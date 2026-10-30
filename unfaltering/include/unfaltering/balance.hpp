// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
#include <chrono>
namespace uf {
struct BackendState {
  const Upstream* target=nullptr;
  std::string name;
  std::size_t active=0;
  bool healthy=true,probing=false;
  unsigned failures=0,recoveries=0,passive_failures=0;
  std::uint64_t transport_errors=0;
  std::chrono::steady_clock::time_point next_probe=std::chrono::steady_clock::now();
  std::uint64_t selections=0;
};
class Balancer {
  const Config& cfg_;
  std::map<std::string,std::size_t> cursors_;
public:
  std::map<std::string,BackendState> backends;
  explicit Balancer(const Config& config);
  BackendState* select(std::string_view target);
  BackendState& state(const Upstream& target);
  bool observe_probe(BackendState& state,bool success);
  bool observe_failure(BackendState& state);
  void observe_success(BackendState& state);
};
}
