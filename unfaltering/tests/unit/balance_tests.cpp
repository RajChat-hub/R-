// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/balance.hpp"
#include "unfaltering/site.hpp"
#include <iostream>
#include <stdexcept>
void check(bool value){if(!value)throw std::runtime_error("pool/health/site invariant");}
int main() {
  uf::Config c;c.upstreams["a"].max_connections=1;c.upstreams["b"].max_connections=1;c.pools["app"].members={"a","b"};
  uf::Balancer balance(c);auto& a=balance.backends.at("a");auto& b=balance.backends.at("b");
  for(unsigned i=0;i<10;++i) {check(balance.select("app")==&a);check(balance.select("app")==&b);}
  a.active=1;check(balance.select("app")==&b);b.active=1;check(!balance.select("app"));a.active=0;b.active=0;
  auto& policy=c.upstreams.at("a").health;policy.enabled=true;policy.failures=2;policy.recoveries=3;
  check(!balance.observe_probe(a,false)&&a.healthy);check(balance.observe_probe(a,false)&&!a.healthy);
  check(!balance.observe_probe(a,true)&&!a.healthy);check(!balance.observe_probe(a,true)&&!a.healthy);
  check(balance.observe_probe(a,true)&&a.healthy);
  check(!balance.observe_failure(a)&&a.healthy);check(balance.observe_failure(a)&&!a.healthy);
  balance.observe_success(a);check(!a.healthy); // App success does not bypass active recovery.
  check(!balance.observe_probe(a,true));check(!balance.observe_probe(a,true));check(balance.observe_probe(a,true));
  uf::ListenerTls sites;sites.default_name="a.test";sites.sites["b.test"]={};sites.unknown_reject=true;sites.require_sni=true;sites.strict_host=true;
  check(uf::select_site(sites,"A.TEST")=="a.test");check(!uf::select_site(sites,"unknown.test"));check(!uf::select_site(sites,""));
  check(uf::site_host_matches(sites,"a.test","a.test"));check(!uf::site_host_matches(sites,"a.test","b.test"));
  for(auto name:{"a..test","*.test","a.test.","127.0.0.1","-bad.test","bad-.test"})check(!uf::dns_identity(name));
  std::cout<<"Deterministic round-robin, active caps, health hysteresis and exact site selection passed\n";
}
