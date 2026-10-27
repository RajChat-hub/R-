// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
namespace uf {
struct Limits {
  std::size_t header_bytes = 16384, header_count = 100, body_bytes = 1048576;
};
struct Request {
  std::string method, target, host, body;
  std::vector<std::pair<std::string, std::string>> headers;
  bool close = false;
};
struct ParseResult {
  enum class State { incomplete, complete, error };
  State state = State::incomplete;
  Request request;
  std::size_t consumed = 0;
  int status = 400;
};
// Incremental bounded parser: incomplete means retain the bytes and append more.
// Total input is bounded by the reactor. No network or filesystem side effects.
ParseResult parse_request(std::string_view input, const Limits &limits = {});
std::string lower(std::string_view input);
bool valid_host(std::string_view host);
std::string reason(int status);
std::string response_headers(int status, std::size_t length, std::string_view mime, bool close,
                             bool allow = false);
} // namespace uf
