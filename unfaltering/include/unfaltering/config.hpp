// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "http.hpp"
#include "routing.hpp"
#include <filesystem>
#include <algorithm>
#include <map>
#include <string>
namespace uf {
struct Certificate {std::filesystem::path certificate,private_key;};
struct ListenerTls {
  std::string default_name;
  std::map<std::string,Certificate> sites;
  bool unknown_reject=false,require_sni=false,strict_host=false;
  bool enabled=false;
  std::filesystem::path certificate,private_key;
  unsigned handshake_ms=3000,shutdown_ms=500;
};
struct Listener {
  std::string name,bind="127.0.0.1";
  unsigned short port=8080;
  unsigned max_connections=0;
  std::string metrics;
  ListenerTls tls;
};
struct Pool {std::vector<std::string> members;};
struct HealthPolicy {
  bool enabled=false;
  std::string path="/health";
  unsigned interval_ms=1000,timeout_ms=500,failures=2,recoveries=2,expect_status=200;
};
struct Upstream {
  HealthPolicy health;
  unsigned max_connections=4096;
  bool tls=false;
  std::string verify_name,sni;
  std::filesystem::path ca_file;
  std::string address,host;
  unsigned short port=0;
  unsigned connect_ms=2000,response_ms=10000,write_ms=5000;
  bool forwarding=true;
};
struct Config {
  ListenerTls tls;
  unsigned probe_concurrency=4,probe_spacing_ms=50;
  std::size_t tls_context_budget_bytes=268435456;
  std::vector<Listener> listeners;
  std::string bind = "127.0.0.1";
  unsigned short port = 8080;
  std::filesystem::path root;
  std::map<std::string, std::filesystem::path> hosts;
  Limits limits;
  std::size_t buffer_bytes = 65536, memory_budget_bytes = 67108864;
  std::vector<Route> routes;
  std::map<std::string,Upstream> upstreams;
  std::map<std::string,Pool> pools;
  std::size_t response_body_bytes=1073741824;
  unsigned max_connections = 256, max_requests = 100, idle_ms = 10000, request_ms = 5000,
           write_ms = 10000, shutdown_ms = 3000;
};
inline std::vector<Listener> configured_listeners(const Config& c) {
  if(!c.listeners.empty())return c.listeners;
  Listener l;l.name="legacy";l.bind=c.bind;l.port=c.port;l.max_connections=c.max_connections;l.tls=c.tls;return {std::move(l)};
}
inline std::size_t probe_reservation(const Config& c) {
  return std::any_of(c.upstreams.begin(),c.upstreams.end(),[](const auto& u){return u.second.health.enabled;})?c.probe_concurrency*393216U:0U;
}
inline bool uses_tls(const Config& c) {
  return c.tls.enabled||std::any_of(c.listeners.begin(),c.listeners.end(),[](const auto& l){return l.tls.enabled;})||
    std::any_of(c.upstreams.begin(),c.upstreams.end(),[](const auto& u){return u.second.tls;});
}
Config parse_config(std::string_view text, const std::filesystem::path &source);
Config read_config(const std::filesystem::path &file);
} // namespace uf
