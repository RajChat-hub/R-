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
ParseResult parse_request(std::string_view in, const Limits &limits) {
  StreamParser parser(limits);
  ParseResult result;
  std::string body;
  while (result.consumed < in.size() && !parser.done() && !parser.failed()) {
    auto n = parser.consume(in.substr(result.consumed), body, limits.body_bytes);
    result.consumed += n;
    if (!n)
      break;
  }
  const auto &h = parser.head();
  result.request.method = h.method;
  result.request.target = h.target;
  result.request.host = h.host;
  result.request.headers = h.headers;
  result.request.close = h.close;
  if (parser.failed()) {
    result.state = ParseResult::State::error;
    result.status = parser.error_status();
  } else if (parser.done()) {
    result.state = ParseResult::State::complete;
    result.request.body = std::move(body);
  } else
    result.consumed = 0;
  return result;
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
  case 502:
    return "Bad Gateway";
  case 504:
    return "Gateway Timeout";
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
         "\r\nServer: UNFALTERING/0.2.0-dev\r\nContent-Length: " + std::to_string(len) +
         "\r\nContent-Type: " + std::string(mime) +
         "\r\nX-Content-Type-Options: nosniff\r\nCache-Control: no-store\r\nConnection: " +
         (close ? "close" : "keep-alive") + "\r\n" + (allow ? "Allow: GET, HEAD\r\n" : "") + "\r\n";
}
} // namespace uf
