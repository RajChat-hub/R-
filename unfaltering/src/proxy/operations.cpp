// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/reactor.hpp"
#include <algorithm>
#include <stdexcept>
#include <sys/epoll.h>
namespace uf {
void Reactor::health_transition(const BackendState& state) {
  ++health_changes;
  log.emit("{\"event\":\"backend_health\",\"backend\":\""+state.name+"\",\"healthy\":"+std::string(state.healthy?"true":"false")+"}\n");
}
void Reactor::finish_probe(std::uint64_t id,bool success,bool observation) {
  auto it=probes.find(id);if(it==probes.end())return;
  auto& probe=*it->second;auto& state=*probe.backend;
  if(observation&&balancer.observe_probe(state,success))health_transition(state);
  if(!success)log.emit("{\"event\":\""+std::string(observation?"probe_failure":"probe_local_error")+"\",\"backend\":\""+state.name+"\",\"tls_code\":"+std::to_string(probe.tls?probe.tls->error:0)+"}\n");
  state.probing=false;state.next_probe=Clock::now()+ms(state.target->health.interval_ms);
  epoll_ctl(epoll.get(),EPOLL_CTL_DEL,probe.socket.get(),nullptr);
  if(success&&probe.tls&&probe.tls->ready())(void)probe.tls->shutdown();
  probes.erase(it);
}
void Reactor::probe_event(std::uint64_t id,std::uint32_t events) {
  auto it=probes.find(id);if(it==probes.end())return;
  int result=-1;
  try {result=it->second->advance(events);}catch(const std::exception&) {result=-1;}
  if(result) {finish_probe(id,result>0);return;}
  epoll_event event{};event.events=it->second->interest();event.data.u64=id;
  if(epoll_ctl(epoll.get(),EPOLL_CTL_MOD,it->second->socket.get(),&event)<0)finish_probe(id,false,false);
}
void Reactor::health_tick() {
  if(stopping)return;
  auto now=Clock::now();
  for(auto it=probes.begin();it!=probes.end();) {
    auto id=it->first;auto* probe=it->second.get();++it;
    if(now>=probe->deadline)finish_probe(id,false);
    else if(probe->tls&&probe->tls->ready()&&probe->tls->pending())probe_event(id,EPOLLIN);
  }
  if(now<next_probe_start||probes.size()>=cfg.probe_concurrency||balancer.backends.empty())return;
  next_probe_start=now+ms(cfg.probe_spacing_ms);
  auto size=balancer.backends.size();
  for(std::size_t tried=0;tried<size;++tried) {
    auto index=probe_cursor++%size;auto it=balancer.backends.begin();std::advance(it,static_cast<std::ptrdiff_t>(index));auto& state=it->second;
    if(!state.target->health.enabled||state.probing||now<state.next_probe)continue;
    try {
      SSL_CTX* context=state.target->tls?tls_upstreams.at(state.name).get():nullptr;
      auto probe=std::make_unique<Probe>(state,context);auto id=next_id++;
      add(probe->socket.get(),id,probe->interest());probes.emplace(id,std::move(probe));state.probing=true;
      ++probes_started;peak_probes=std::max(peak_probes,probes.size());
    } catch(const std::exception&) {
      state.probing=false;state.next_probe=now+ms(state.target->health.interval_ms);
      log.emit("{\"event\":\"probe_local_error\",\"backend\":\""+state.name+"\"}\n");
    }
    break; // Global minimum spacing prevents a failure-driven probe burst.
  }
}
void Reactor::metrics(Connection& c) {
  // Loopback binding alone does not prevent browser DNS rebinding. Do not
  // expose topology to arbitrary HTTP authorities resolving to localhost.
  const auto& host=c.parser.head().host;
  if(host!="localhost"&&host!=listeners[c.listener_index].config.bind) {c.close=true;prepare(c,403);return;}
  if(c.parser.head().method!="GET"&&c.parser.head().method!="HEAD") {prepare(c,405);return;}
  std::string body;
  auto append=[&](const std::string& value) {
    if(value.size()>cfg.buffer_bytes-1024-body.size())throw std::length_error("metrics response budget");
    body+=value;
  };
  try {
    append("{\"active_clients\":"+std::to_string(clients.size())+",\"active_probes\":"+std::to_string(probes.size())+",\"peak_probes\":"+std::to_string(peak_probes)+",\"probes_started\":"+std::to_string(probes_started)+",\"selection_failures\":"+std::to_string(selection_failures)+",\"health_changes\":"+std::to_string(health_changes)+",\"listeners\":[");
    bool comma=false;
    for(const auto& listener:listeners) {
      append(std::string(comma?",":"")+"{\"name\":\""+listener.config.name+"\",\"active\":"+std::to_string(listener.active)+"}");comma=true;
    }
    append("],\"backends\":[");comma=false;
    for(const auto& [name,state]:balancer.backends) {
      append(std::string(comma?",":"")+"{\"name\":\""+name+"\",\"healthy\":"+std::string(state.healthy?"true":"false")+",\"active\":"+std::to_string(state.active)+",\"probing\":"+std::string(state.probing?"true":"false")+",\"failures\":"+std::to_string(state.failures)+",\"recoveries\":"+std::to_string(state.recoveries)+",\"transport_errors\":"+std::to_string(state.transport_errors)+",\"selections\":"+std::to_string(state.selections)+"}");comma=true;
    }
    append("],\"pools\":[");comma=false;
    for(const auto& [name,pool]:cfg.pools) {
      std::size_t healthy=0,active=0;
      for(const auto& member:pool.members) {const auto& state=balancer.backends.at(member);healthy+=state.healthy?1U:0U;active+=state.active;}
      append(std::string(comma?",":"")+"{\"name\":\""+name+"\",\"members\":"+std::to_string(pool.members.size())+",\"healthy_members\":"+std::to_string(healthy)+",\"active\":"+std::to_string(active)+"}");comma=true;
    }
    append("]}\n");
  } catch(const std::length_error&) {c.close=true;prepare(c,503);return;}
  c.output=response_headers(200,body.size(),"application/json",c.close);
  if(c.parser.head().method!="HEAD")c.output+=body;
  c.sent=0;c.file.reset();c.file_sent=c.file_length=0;
}
}
