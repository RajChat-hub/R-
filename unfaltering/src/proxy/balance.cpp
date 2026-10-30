// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/balance.hpp"
#include <stdexcept>
namespace uf {
Balancer::Balancer(const Config& cfg):cfg_(cfg) {
  for(const auto& [name,target]:cfg.upstreams) {
    BackendState state;state.target=&target;state.name=name;backends.emplace(name,std::move(state));
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
}
