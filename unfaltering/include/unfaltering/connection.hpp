// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
#include "fd.hpp"
#include "proxy.hpp"
#include "tls.hpp"
#include <memory>
#include <algorithm>
#include <chrono>
#include <cstdint>
namespace uf {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
inline auto ms(unsigned value) {
  return std::chrono::milliseconds(value);
}
struct Connection {
  Fd socket, file;
  std::size_t listener_index=0;
  unsigned tls_shutdown_ms=500;
  std::unique_ptr<Tls> tls;
  bool tls_closing=false,graceful=false;
  Time tls_deadline{};
  StreamParser parser;
  std::string input, output;
  std::size_t sent = 0, file_sent = 0, file_length = 0;
  Time deadline = Clock::now();
  bool started = false, writing = false, close = false, eof = false, io_pending = false;
  std::unique_ptr<ProxyTransaction> proxy;
  std::string peer;
  bool committed=false,dead=false;
  Time upload_deadline{},write_deadline{};
  unsigned requests = 0;
  std::uint64_t epoch = 0;
  explicit Connection(int fd, const Config &cfg)
      : socket(fd), parser(cfg.limits), deadline(Clock::now() + ms(cfg.idle_ms)) {}
};
inline std::size_t connection_reservation(const Config &c) {
  return 8 * c.limits.header_bytes + 8 * c.buffer_bytes + 512 * c.limits.header_count + 32768 +
    (uses_tls(c)?524288U:0U);
}
} // namespace uf
