// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/config.hpp"
#include <arpa/inet.h>
#include <charconv>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>
namespace uf {
namespace {
struct Lex {
  std::string input;
  std::size_t pos = 0, line = 1;
  std::string word;
  [[noreturn]] void error(const std::string &message) const {
    throw std::runtime_error("config line " + std::to_string(line) + ": " + message);
  }
  void next() {
    word.clear();
    while (pos < input.size()) {
      char c = input[pos];
      if (c == '#') {
        while (pos < input.size() && input[pos] != '\n')
          ++pos;
        continue;
      }
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        if (c == '\n')
          ++line;
        ++pos;
        continue;
      }
      break;
    }
    if (pos == input.size())
      return;
    char c = input[pos++];
    if (c == '{' || c == '}' || c == ';') {
      word = c;
      return;
    }
    if (c == '"') {
      while (pos < input.size() && input[pos] != '"') {
        if (input[pos] == '\n' || input[pos] == '\r' || input[pos] == '\\' ||
            static_cast<unsigned char>(input[pos]) < 32)
          error("invalid quoted value");
        word += input[pos++];
      }
      if (pos == input.size())
        error("unterminated quote");
      ++pos;
      return;
    }
    word += c;
    while (pos < input.size()) {
      c = input[pos];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ';' || c == '{' || c == '}' ||
          c == '#')
        break;
      word += c;
      ++pos;
    }
  }
  void need(const std::string &s) {
    if (word != s)
      error("expected '" + s + "', found '" + word + "'");
    next();
  }
  std::string value() {
    if (word.empty() || word == "{" || word == "}" || word == ";")
      error("expected value");
    auto v = word;
    next();
    need(";");
    return v;
  }
};
unsigned numeric(Lex &l, unsigned min, unsigned max) {
  auto v = l.value();
  unsigned n = 0;
  auto [p, e] = std::from_chars(v.data(), v.data() + v.size(), n);
  if (e != std::errc{} || p != v.data() + v.size() || n < min || n > max)
    l.error("numeric limit out of range");
  return n;
}
} // namespace
Config read_config(const std::filesystem::path &file) {
  std::ifstream f(file, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open config");
  std::string text;
  char buffer[4096];
  while (f.read(buffer, sizeof buffer) || f.gcount()) {
    text.append(buffer, static_cast<std::size_t>(f.gcount()));
    if (text.size() > 65536)
      throw std::runtime_error("config exceeds 64 KiB");
  }
  return parse_config(text, file);
}
Config parse_config(std::string_view text, const std::filesystem::path &file) {
  if (text.size() > 65536)
    throw std::runtime_error("config exceeds 64 KiB");
  Lex l{std::string(text), 0, 1, {}};
  Config c;
  std::set<std::string> keys;
  l.next();
  l.need("server");
  l.need("{");
  auto path = [&](const std::string &s) {
    auto p = std::filesystem::path(s);
    if (p.is_relative())
      p = std::filesystem::absolute(file).parent_path() / p;
    if (!std::filesystem::is_directory(p))
      l.error("root is not a directory");
    return p.lexically_normal();
  };
  while (l.word != "}") {
    auto key = l.word;
    if (key.empty())
      l.error("missing closing brace");
    l.next();
    if (key == "host") {
      auto name = lower(l.word);
      if (!valid_host(name) || name.find(':') != name.npos || c.hosts.contains(name))
        l.error("invalid or duplicate virtual host");
      l.next();
      l.need("{");
      l.need("root");
      auto p = path(l.value());
      l.need("}");
      c.hosts.emplace(name, p);
      continue;
    }
    if (key == "route") {
      bool exact = l.word == "exact";
      if (!exact && l.word != "prefix")
        l.error("route requires exact or prefix");
      l.next();
      auto target = l.word;
      if (target.empty() || target.front() != '/' || target.find_first_of("%?#\\") != target.npos)
        l.error("route requires a literal absolute path");
      for (const auto &route : c.routes)
        if (route.exact == exact && route.path == target)
          l.error("duplicate route");
      l.next();
      l.need("{");
      l.need("root");
      auto p = path(l.value());
      l.need("}");
      c.routes.push_back(Route{exact, target, p});
      continue;
    }
    if (!keys.insert(key).second)
      l.error("duplicate key '" + key + "'");
    if (key == "listen") {
      auto address = l.value();
      auto colon = address.find(':');
      if (colon == address.npos)
        l.error("expected numeric IPv4:port");
      c.bind = address.substr(0, colon);
      in_addr ip{};
      if (inet_pton(AF_INET, c.bind.c_str(), &ip) != 1)
        l.error("listen requires numeric IPv4");
      auto s = address.substr(colon + 1);
      unsigned n = 0;
      auto [p, e] = std::from_chars(s.data(), s.data() + s.size(), n);
      if (e != std::errc{} || p != s.data() + s.size() || n < 1 || n > 65535)
        l.error("invalid listen port");
      c.port = static_cast<unsigned short>(n);
    } else if (key == "root")
      c.root = path(l.value());
    else if (key == "max_connections")
      c.max_connections = numeric(l, 1, 4096);
    else if (key == "max_requests")
      c.max_requests = numeric(l, 1, 10000);
    else if (key == "header_bytes")
      c.limits.header_bytes = numeric(l, 256, 65536);
    else if (key == "header_count")
      c.limits.header_count = numeric(l, 1, 200);
    else if (key == "body_bytes")
      c.limits.body_bytes = numeric(l, 0, 1048576);
    else if (key == "idle_ms")
      c.idle_ms = numeric(l, 100, 60000);
    else if (key == "request_ms")
      c.request_ms = numeric(l, 100, 60000);
    else if (key == "write_ms")
      c.write_ms = numeric(l, 100, 60000);
    else if (key == "shutdown_ms")
      c.shutdown_ms = numeric(l, 100, 30000);
    else
      l.error("unknown key '" + key + "'");
  }
  l.need("}");
  if (!l.word.empty())
    l.error("trailing content");
  if (!keys.contains("listen") || c.root.empty())
    l.error("listen and root are required");
  return c;
}
} // namespace uf
