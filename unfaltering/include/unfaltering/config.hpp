// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "http.hpp"
#include "routing.hpp"
#include <filesystem>
#include <map>
#include <string>
namespace uf {
struct Config {
  std::string bind = "127.0.0.1";
  unsigned short port = 8080;
  std::filesystem::path root;
  std::map<std::string, std::filesystem::path> hosts;
  Limits limits;
  std::vector<Route> routes;
  unsigned max_connections = 256, max_requests = 100, idle_ms = 10000, request_ms = 5000,
           write_ms = 10000, shutdown_ms = 3000;
};
Config parse_config(std::string_view text, const std::filesystem::path &source);
Config read_config(const std::filesystem::path &file);
} // namespace uf
