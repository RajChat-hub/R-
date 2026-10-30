// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
namespace uf {
struct BackendState {
  const Upstream* target=nullptr;
  std::string name;
  std::size_t active=0;
  bool healthy=true;
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
};
}
