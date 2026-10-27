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
enum class Framing { none, fixed, chunked, eof };
struct MessageHead {
  std::string method, target, host, authority, phrase;
  std::vector<std::pair<std::string, std::string>> headers;
  int status = 0;
  bool close = false;
  Framing framing = Framing::none;
  std::size_t length = 0;
};
// Consumes each wire byte once, retaining only bounded metadata. Decoded payload
// goes to the caller's bounded sink. A header boundary is always returned first.
class StreamParser {
public:
  enum class Stage {
    first,
    headers,
    fixed,
    chunk_size,
    chunk_data,
    chunk_crlf,
    trailers,
    eof_body,
    done,
    failed
  };

private:
  Limits limits_;
  MessageHead head_;
  Stage stage_ = Stage::first;
  std::string line_;
  std::size_t header_bytes_ = 0, header_count_ = 0, remaining_ = 0, decoded_ = 0, wire_ = 0;
  unsigned crlf_ = 0;
  bool response_ = false, head_response_ = false, ready_ = false, host_ = false, length_ = false,
       te_ = false;
  int error_ = 400;
  void fail(int code);
  void line();
  void finish_headers();

public:
  explicit StreamParser(Limits limits = {}, bool response = false, bool head_response = false);
  std::size_t consume(std::string_view bytes, std::string &sink, std::size_t capacity);
  void eof();
  bool ready() const {
    return ready_;
  }
  bool done() const {
    return stage_ == Stage::done;
  }
  bool failed() const {
    return stage_ == Stage::failed;
  }
  int error_status() const {
    return error_;
  }
  const MessageHead &head() const {
    return head_;
  }
  std::size_t decoded_bytes() const {
    return decoded_;
  }
  std::size_t wire_bytes() const {
    return wire_;
  }
  Stage stage() const {
    return stage_;
  }
};
// Compatibility one-shot helper for existing tests/tools; the reactor owns and
// reuses a StreamParser instead of repeatedly invoking this helper.
ParseResult parse_request(std::string_view input, const Limits &limits = {});
std::string lower(std::string_view input);
bool valid_host(std::string_view host);
std::string reason(int status);
std::string response_headers(int status, std::size_t length, std::string_view mime, bool close,
                             bool allow = false);
} // namespace uf
