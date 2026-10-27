// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/http.hpp"
#include <algorithm>
#include <charconv>
#include <limits>
namespace uf {
namespace {
bool token(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
         std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}
std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    s.remove_suffix(1);
  return s;
}
} // namespace
StreamParser::StreamParser(Limits limits, bool response, bool head_response)
    : limits_(limits), response_(response), head_response_(head_response) {}
void StreamParser::fail(int code) {
  error_ = response_ ? 502 : code;
  stage_ = Stage::failed;
}
void StreamParser::finish_headers() {
  if ((!response_ && !host_) || (length_ && te_)) {
    fail(400);
    return;
  }
  bool empty = response_ &&
               (head_response_ || head_.status == 204 || head_.status == 304 || head_.status < 200);
  if (response_ && (head_.status == 204 || head_.status < 200) && (te_ || length_)) {
    fail(400);
    return;
  }
  if (empty) {
    head_.framing = Framing::none;
    stage_ = Stage::done;
  } else if (te_) {
    head_.framing = Framing::chunked;
    stage_ = Stage::chunk_size;
  } else if (length_) {
    if (head_.length > limits_.body_bytes) {
      fail(413);
      return;
    }
    head_.framing = Framing::fixed;
    remaining_ = head_.length;
    stage_ = remaining_ ? Stage::fixed : Stage::done;
  } else if (response_) {
    head_.framing = Framing::eof;
    stage_ = Stage::eof_body;
  } else {
    head_.framing = Framing::none;
    stage_ = Stage::done;
  }
  ready_ = true;
}
void StreamParser::line() {
  auto s = std::string_view(line_).substr(0, line_.size() - 2);
  if (stage_ == Stage::first) {
    if (response_) {
      if (s.size() < 13 || (s.substr(0, 9) != "HTTP/1.1 " && s.substr(0, 9) != "HTTP/1.0 ")) {
        fail(400);
        return;
      }
      auto code = s.substr(9, 3);
      int n = 0;
      auto [p, e] = std::from_chars(code.data(), code.data() + 3, n);
      if (e != std::errc{} || p != code.data() + 3 || n < 100 || n > 599 || n == 101) {
        fail(400);
        return;
      }
      // Status line includes a required SP after its three-digit code.
      if (s.size() < 13 || s[12] != ' ') {
        fail(400);
        return;
      }
      head_.status = n;
      head_.phrase = s.substr(13);
      for (char raw : head_.phrase) {
        auto c = static_cast<unsigned char>(raw);
        if (c < 32 || c == 127) {
          fail(400);
          return;
        }
      }
    } else {
      auto a = s.find(' '), b = s.find(' ', a == s.npos ? 0 : a + 1);
      if (a == s.npos || b == s.npos || s.find(' ', b + 1) != s.npos) {
        fail(400);
        return;
      }
      auto method = s.substr(0, a), target = s.substr(a + 1, b - a - 1);
      if (method.empty() || !std::all_of(method.begin(), method.end(), token)) {
        fail(400);
        return;
      }
      head_.method = method;
      if (s.substr(b + 1) != "HTTP/1.1") {
        fail(505);
        return;
      }
      if (target.empty() || target.front() != '/' || target.find('#') != target.npos) {
        fail(400);
        return;
      }
      for (char raw : target) {
        auto c = static_cast<unsigned char>(raw);
        if (c <= 32 || c >= 127) {
          fail(400);
          return;
        }
      }
      head_.target = target;
    }
    stage_ = Stage::headers;
  } else if (stage_ == Stage::headers) {
    if (s.empty()) {
      finish_headers();
      return;
    }
    if (++header_count_ > limits_.header_count) {
      fail(431);
      return;
    }
    auto colon = s.find(':');
    if (colon == s.npos || !colon ||
        !std::all_of(s.begin(), s.begin() + static_cast<std::ptrdiff_t>(colon), token)) {
      fail(400);
      return;
    }
    auto name = lower(s.substr(0, colon));
    auto value = trim(s.substr(colon + 1));
    for (char raw : value) {
      auto c = static_cast<unsigned char>(raw);
      if ((c < 32 && c != 9) || c == 127) {
        fail(400);
        return;
      }
    }
    head_.headers.emplace_back(name, value);
    if (!response_ && name == "host") {
      if (host_ || !valid_host(value)) {
        fail(400);
        return;
      }
      host_ = true;
      head_.authority = value;
      head_.host = lower(value.substr(0, value.find(':')));
    }
    if (name == "content-length") {
      if (length_ || value.empty()) {
        fail(400);
        return;
      }
      length_ = true;
      auto [p, e] = std::from_chars(value.data(), value.data() + value.size(), head_.length);
      if (e != std::errc{} || p != value.data() + value.size()) {
        fail(400);
        return;
      }
    }
    if (name == "transfer-encoding") {
      if (te_ || lower(value) != "chunked") {
        fail(400);
        return;
      }
      te_ = true;
    }
    if (!response_ && name == "expect") {
      fail(417);
      return;
    }
    if (name == "connection") {
      if (value.empty()) {
        fail(400);
        return;
      }
      for (;;) {
        auto comma = value.find(',');
        if(++connection_count_>limits_.header_count) {fail(431);return;}
        auto t = trim(value.substr(0, comma));
        if (t.empty() || !std::all_of(t.begin(), t.end(), token)) {
          fail(400);
          return;
        }
        if (lower(t) == "close")
          head_.close = true;
        if (comma == value.npos)
          break;
        value.remove_prefix(comma + 1);
      }
    }
  } else if (stage_ == Stage::chunk_size) {
    if (s.empty() || s.size() > 16) {
      fail(400);
      return;
    }
    auto [p, e] = std::from_chars(s.data(), s.data() + s.size(), remaining_, 16);
    if (e != std::errc{} || p != s.data() + s.size()) {
      fail(400);
      return;
    }
    if (remaining_ > limits_.body_bytes - decoded_) {
      fail(413);
      return;
    }
    stage_ = remaining_ ? Stage::chunk_data : Stage::trailers;
  } else if (stage_ == Stage::trailers) {
    if (!s.empty()) {
      fail(400);
      return;
    }
    stage_ = Stage::done;
  }
}
std::size_t StreamParser::consume(std::string_view bytes, std::string &sink, std::size_t capacity) {
  std::size_t i = 0;
  while (i < bytes.size() && !done() && !failed()) {
    if (stage_ == Stage::fixed || stage_ == Stage::chunk_data || stage_ == Stage::eof_body) {
      auto available = capacity > sink.size() ? capacity - sink.size() : 0;
      if (!available)
        break;
      auto n = std::min(bytes.size() - i, available);
      if (stage_ != Stage::eof_body)
        n = std::min(n, remaining_);
      if (n > limits_.body_bytes - decoded_) {
        fail(413);
        break;
      }
      sink.append(bytes.substr(i, n));
      i += n;
      decoded_ += n;
      wire_ += n;
      if (stage_ != Stage::eof_body) {
        remaining_ -= n;
        if (!remaining_) {
          stage_ = stage_ == Stage::fixed ? Stage::done : Stage::chunk_crlf;
          crlf_ = 0;
        }
      }
      continue;
    }
    char c = bytes[i++];
    ++wire_;
    if (stage_ == Stage::chunk_crlf) {
      if (c != (crlf_ == 0 ? '\r' : '\n')) {
        fail(400);
        break;
      }
      if (++crlf_ == 2)
        stage_ = Stage::chunk_size;
      continue;
    }
    bool header = stage_ == Stage::first || stage_ == Stage::headers;
    if (header && ++header_bytes_ > limits_.header_bytes) {
      fail(431);
      break;
    }
    if (!line_.empty() && line_.back() == '\r' && c != '\n') {
      fail(400);
      break;
    }
    line_ += c;
    if (c == '\n') {
      if (line_.size() < 2 || line_[line_.size() - 2] != '\r') {
        fail(400);
        break;
      }
      bool was_ready = ready_;
      line();
      line_.clear();
      if (!was_ready && ready_)
        break;
    } else if (!header && line_.size() > 18) {
      fail(400);
      break;
    }
  }
  return i;
}
void StreamParser::eof() {
  if (stage_ == Stage::eof_body)
    stage_ = Stage::done;
  else if (!done())
    fail(400);
}
} // namespace uf
