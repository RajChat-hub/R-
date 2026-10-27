// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/http.hpp"
#include <algorithm>
#include <cctype>
#include <charconv>
#include <ctime>
#include <limits>
namespace uf {
std::string lower(std::string_view s) {
  std::string o(s);
  for (char &c : o)
    if (c >= 'A' && c <= 'Z')
      c = static_cast<char>(c - 'A' + 'a');
  return o;
}
static bool token(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
         std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}
static std::string_view trim(std::string_view s) {
  while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
    s.remove_prefix(1);
  while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
    s.remove_suffix(1);
  return s;
}
bool valid_host(std::string_view h) {
  if (h.empty() || h.size() > 259)
    return false;
  // This initial listener is IPv4-only; reject bracketed IPv6 rather than misparse it.
  auto p = h.find(':');
  auto name = h.substr(0, p);
  if (name.empty() || name.front() == '.' || name.back() == '.')
    return false;
  std::size_t label = 0;
  for (char c : name) {
    if (c == '.') {
      if (label == 0 || label > 63)
        return false;
      label = 0;
    } else {
      if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-'))
        return false;
      ++label;
    }
  }
  if (label == 0 || label > 63)
    return false;
  if (p != h.npos) {
    unsigned port = 0;
    auto s = h.substr(p + 1);
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), port);
    if (s.empty() || ec != std::errc{} || end != s.data() + s.size() || port == 0 || port > 65535)
      return false;
  }
  return true;
}
ParseResult parse_request(std::string_view in, const Limits &lim) {
  ParseResult out;
  auto fail = [&](int status) {
    out.state = ParseResult::State::error;
    out.status = status;
    return out;
  };
  auto end = in.find("\r\n\r\n");
  if (end == in.npos) {
    if (in.size() > lim.header_bytes)
      return fail(431);
    return out;
  }
  if (end + 4 > lim.header_bytes)
    return fail(431);
  auto line_end = in.find("\r\n");
  if (line_end == in.npos)
    return fail(400);
  auto first = in.substr(0, line_end);
  auto a = first.find(' '), b = first.find(' ', a == first.npos ? 0 : a + 1);
  if (a == first.npos || b == first.npos || first.find(' ', b + 1) != first.npos)
    return fail(400);
  auto method = first.substr(0, a), target = first.substr(a + 1, b - a - 1),
       version = first.substr(b + 1);
  if (method.empty() || !std::all_of(method.begin(), method.end(), token))
    return fail(400);
  if (version != "HTTP/1.1")
    return fail(505);
  if (target.empty() || target.front() != '/' || target.find('#') != target.npos)
    return fail(400);
  for (char raw : target) {
    auto c = static_cast<unsigned char>(raw);
    if (c <= 32 || c >= 127)
      return fail(400);
  }
  out.request.method = method;
  out.request.target = target;
  bool have_host = false, have_length = false, have_te = false;
  std::size_t length = 0, count = 0;
  std::size_t pos = line_end + 2;
  while (pos < end) {
    auto e = in.find("\r\n", pos);
    if (e == in.npos || e > end)
      return fail(400);
    auto line = in.substr(pos, e - pos);
    pos = e + 2;
    auto colon = line.find(':');
    if (++count > lim.header_count)
      return fail(431);
    if (colon == line.npos || colon == 0 ||
        !std::all_of(line.begin(), line.begin() + static_cast<std::ptrdiff_t>(colon), token))
      return fail(400);
    auto name = lower(line.substr(0, colon));
    auto value = trim(line.substr(colon + 1));
    for (char raw : value) {
      auto c = static_cast<unsigned char>(raw);
      if ((c < 32 && c != 9) || c == 127)
        return fail(400);
    }
    out.request.headers.emplace_back(name, value);
    if (name == "host") {
      if (have_host || !valid_host(value))
        return fail(400);
      have_host = true;
      out.request.host = lower(value.substr(0, value.find(':')));
    }
    if (name == "content-length") {
      // Deliberately stricter than RFC's optional recovery: reject every duplicate.
      if (have_length || value.empty())
        return fail(400);
      have_length = true;
      auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), length);
      if (ec != std::errc{} || ptr != value.data() + value.size())
        return fail(400);
      if (length > lim.body_bytes)
        return fail(413);
    }
    if (name == "transfer-encoding") {
      if (have_te || lower(value) != "chunked")
        return fail(400);
      have_te = true;
    }
    if (name == "expect")
      return fail(417); // 100-continue deliberately unsupported.
    if (name == "connection") {
      while (!value.empty()) {
        auto comma = value.find(',');
        auto t = trim(value.substr(0, comma));
        if (t.empty() || !std::all_of(t.begin(), t.end(), token))
          return fail(400);
        if (lower(t) == "close")
          out.request.close = true;
        if (comma == value.npos)
          break;
        value.remove_prefix(comma + 1);
      }
    }
  }
  if (!have_host || (have_te && have_length))
    return fail(400);
  pos = end + 4;
  if (have_te) {
    // Strict chunked subset: chunk extensions and nonempty trailers are rejected.
    // Their total framing overhead is bounded by the reactor input limit.
    std::string body;
    for (;;) {
      auto e = in.find("\r\n", pos);
      if (e == in.npos) {
        if (in.size() - pos > 32)
          return fail(400);
        return out;
      }
      auto size = in.substr(pos, e - pos);
      if (size.empty() || size.size() > 16)
        return fail(400);
      std::size_t n = 0;
      auto [ptr, ec] = std::from_chars(size.data(), size.data() + size.size(), n, 16);
      if (ec != std::errc{} || ptr != size.data() + size.size())
        return fail(400);
      if (n > lim.body_bytes - body.size())
        return fail(413);
      pos = e + 2;
      if (n == 0) {
        if (in.size() - pos < 2)
          return out;
        if (in.substr(pos, 2) != "\r\n")
          return fail(400);
        pos += 2;
        break;
      }
      if (in.size() - pos < n + 2)
        return out;
      if (in.substr(pos + n, 2) != "\r\n")
        return fail(400);
      body.append(in.substr(pos, n));
      pos += n + 2;
    }
    out.request.body = std::move(body);
  } else {
    if (in.size() - pos < length)
      return out;
    out.request.body = in.substr(pos, length);
    pos += length;
  }
  out.consumed = pos;
  out.state = ParseResult::State::complete;
  return out;
}
std::string reason(int s) {
  switch (s) {
  case 200:
    return "OK";
  case 400:
    return "Bad Request";
  case 403:
    return "Forbidden";
  case 404:
    return "Not Found";
  case 405:
    return "Method Not Allowed";
  case 408:
    return "Request Timeout";
  case 413:
    return "Content Too Large";
  case 417:
    return "Expectation Failed";
  case 431:
    return "Request Header Fields Too Large";
  case 500:
    return "Internal Server Error";
  case 503:
    return "Service Unavailable";
  case 505:
    return "HTTP Version Not Supported";
  default:
    return "Error";
  }
}
std::string response_headers(int status, std::size_t len, std::string_view mime, bool close,
                             bool allow) {
  std::time_t now = std::time(nullptr);
  std::tm utc{};
  gmtime_r(&now, &utc);
  char date[64]{};
  std::strftime(date, sizeof date, "%a, %d %b %Y %H:%M:%S GMT", &utc);
  return "HTTP/1.1 " + std::to_string(status) + " " + reason(status) +
         "\r\nDate: " + std::string(date) +
         "\r\nServer: UNFALTERING/0.1.0-dev\r\nContent-Length: " + std::to_string(len) +
         "\r\nContent-Type: " + std::string(mime) +
         "\r\nX-Content-Type-Options: nosniff\r\nCache-Control: no-store\r\nConnection: " +
         (close ? "close" : "keep-alive") + "\r\n" + (allow ? "Allow: GET, HEAD\r\n" : "") + "\r\n";
}
} // namespace uf
