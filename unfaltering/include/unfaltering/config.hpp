// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "http.hpp"
#include "routing.hpp"
#include <filesystem>
#include <algorithm>
#include <map>
#include <string>
namespace uf {
struct ListenerTls {
  bool enabled=false;
  std::filesystem::path certificate,private_key;
  unsigned handshake_ms=3000,shutdown_ms=500;
};
struct Listener {
  std::string name,bind="127.0.0.1";
  unsigned short port=8080;
  unsigned max_connections=0;
  ListenerTls tls;
};
struct Upstream {
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
  std::vector<Listener> listeners;
  std::string bind = "127.0.0.1";
  unsigned short port = 8080;
  std::filesystem::path root;
  std::map<std::string, std::filesystem::path> hosts;
  Limits limits;
  std::size_t buffer_bytes = 65536, memory_budget_bytes = 67108864;
  std::vector<Route> routes;
  std::map<std::string,Upstream> upstreams;
  std::size_t response_body_bytes=1073741824;
  unsigned max_connections = 256, max_requests = 100, idle_ms = 10000, request_ms = 5000,
           write_ms = 10000, shutdown_ms = 3000;
};
inline std::vector<Listener> configured_listeners(const Config& c) {
  if(!c.listeners.empty())return c.listeners;
  return {Listener{"legacy",c.bind,c.port,c.max_connections,c.tls}};
}
inline bool uses_tls(const Config& c) {
  return c.tls.enabled||std::any_of(c.listeners.begin(),c.listeners.end(),[](const auto& l){return l.tls.enabled;})||
    std::any_of(c.upstreams.begin(),c.upstreams.end(),[](const auto& u){return u.second.tls;});
}
Config parse_config(std::string_view text, const std::filesystem::path &source);
Config read_config(const std::filesystem::path &file);
} // namespace uf
