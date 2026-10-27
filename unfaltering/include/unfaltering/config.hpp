// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "http.hpp"
#include "routing.hpp"
#include <filesystem>
#include <map>
#include <string>
namespace uf {
struct Upstream {
  std::string address,host;
  unsigned short port=0;
  unsigned connect_ms=2000,response_ms=10000,write_ms=5000;
  bool forwarding=true;
};
struct Config {
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
Config parse_config(std::string_view text, const std::filesystem::path &source);
Config read_config(const std::filesystem::path &file);
} // namespace uf
