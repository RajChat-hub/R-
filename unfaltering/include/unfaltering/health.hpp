// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "balance.hpp"
#include "fd.hpp"
#include "tls.hpp"
#include <chrono>
namespace uf {
class Probe {
  std::string outgoing_,incoming_;
  std::size_t sent_=0;
  StreamParser parser_{Limits{8192,32,65536},true};
  unsigned informational_=0;
  bool connecting_=true,failed_=false;
  int consume();
public:
  BackendState* backend;
  Fd socket;
  std::unique_ptr<Tls> tls;
  std::chrono::steady_clock::time_point deadline;
  Probe(BackendState& state,SSL_CTX* context);
  std::uint32_t interest() const;
  int advance(std::uint32_t events); // 0 pending, 1 healthy complete message, -1 failure.
};
}
