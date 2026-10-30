// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/balance.hpp"
#include <stdexcept>
namespace uf {
Balancer::Balancer(const Config& cfg):cfg_(cfg) {
  for(const auto& [name,target]:cfg.upstreams) {
    BackendState state;state.target=&target;state.name=name;state.healthy=!target.health.enabled;backends.emplace(name,std::move(state));
  }
  for(const auto& [name,pool]:cfg.pools) {(void)pool;cursors_.emplace(name,0);}
}
BackendState& Balancer::state(const Upstream& target) {
  for(auto& [name,state]:backends) { (void)name;if(state.target==&target)return state; }
  throw std::runtime_error("backend identity invariant");
}
BackendState* Balancer::select(std::string_view target) {
  auto eligible=[](const BackendState& backend) {return backend.healthy&&backend.active<backend.target->max_connections;};
  auto pool=cfg_.pools.find(std::string(target));
  if(pool==cfg_.pools.end()) {
    auto it=backends.find(std::string(target));
    if(it==backends.end()||!eligible(it->second))return nullptr;
    ++it->second.selections;return &it->second;
  }
  auto& cursor=cursors_.at(pool->first);const auto& members=pool->second.members;
  for(std::size_t i=0;i<members.size();++i) {
    auto index=(cursor+i)%members.size();auto& backend=backends.at(members[index]);
    if(eligible(backend)) {cursor=(index+1)%members.size();++backend.selections;return &backend;}
  }
  return nullptr;
}
bool Balancer::observe_probe(BackendState& state,bool success) {
  auto old=state.healthy;const auto& policy=state.target->health;
  if(success) {
    state.failures=0;state.passive_failures=0;
    state.recoveries=std::min(state.recoveries+1,policy.recoveries);
    if(state.recoveries>=policy.recoveries)state.healthy=true;
  } else {
    state.recoveries=0;state.failures=std::min(state.failures+1,policy.failures);
    if(state.failures>=policy.failures)state.healthy=false;
  }
  return old!=state.healthy;
}
bool Balancer::observe_failure(BackendState& state) {
  ++state.transport_errors;
  if(!state.target->health.enabled)return false;
  auto old=state.healthy;state.recoveries=0;
  state.passive_failures=std::min(state.passive_failures+1,state.target->health.failures);
  if(state.passive_failures>=state.target->health.failures)state.healthy=false;
  return old!=state.healthy;
}
void Balancer::observe_success(BackendState& state) {state.passive_failures=0;}

}
