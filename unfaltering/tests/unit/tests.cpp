// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/config.hpp"
#include "unfaltering/http.hpp"
#include "unfaltering/static.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
namespace {
unsigned checks = 0;
void expect(bool ok, const std::string &name) {
  ++checks;
  if (!ok)
    throw std::runtime_error(name);
}
std::string req(std::string headers = "Host: example.test\r\n", std::string body = "") {
  return "GET / HTTP/1.1\r\n" + headers + "\r\n" + body;
}
} // namespace
int main() {
  try {
    using S = uf::ParseResult::State;
    const auto base = req();
    for (std::size_t i = 0; i < base.size(); ++i)
      expect(uf::parse_request(base.substr(0, i)).state == S::incomplete,
             "fragment prefix " + std::to_string(i));
    auto p = uf::parse_request(base + base);
    expect(p.state == S::complete && p.consumed == base.size(), "pipeline framing");
    expect(p.request.host == "example.test", "host normalized");
    for (const auto &header : std::vector<std::string>{
             "", "Host: x\r\nHost: y\r\n", "Host: x\r\nContent-Length: 1\r\nContent-Length: 1\r\n",
             "Host: x\r\nContent-Length: -1\r\n", "Host: x\r\nContent-Length: 1, 1\r\n",
             "Host: x\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n",
             "Host: x\r\nTransfer-Encoding: gzip, chunked\r\n", "Host: x\r\n Bad: value\r\n",
             "Host : x\r\n", "Host: x\r\nX: a\nb\r\n", "Host: x/y\r\n", "Host: x:99999\r\n"})
      expect(uf::parse_request(req(header)).state == S::error,
             "reject ambiguous or invalid header");
    p = uf::parse_request(req("Host: x\r\nContent-Length: 3\r\n", "abc"));
    expect(p.state == S::complete && p.request.body == "abc", "fixed length");
    expect(uf::parse_request(req("Host: x\r\nContent-Length: 3\r\n", "ab")).state == S::incomplete,
           "partial body");
    const auto chunk =
        req("Host: x\r\nTransfer-Encoding: chunked\r\n", "3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n");
    for (std::size_t i = 0; i < chunk.size(); ++i)
      expect(uf::parse_request(chunk.substr(0, i)).state == S::incomplete, "fragmented chunks");
    p = uf::parse_request(chunk);
    expect(p.state == S::complete && p.request.body == "abcde", "chunk decode");
    for (const auto &body :
         std::vector<std::string>{"g\r\n", "1;x=y\r\na\r\n0\r\n\r\n", "1\r\naX\r\n",
                                  "0\r\nHost: bad\r\n\r\n", "ffffffffffffffff\r\n"})
      expect(uf::parse_request(req("Host: x\r\nTransfer-Encoding: chunked\r\n", body)).state ==
                 S::error,
             "bad chunk");
    uf::Limits l;
    l.body_bytes = 2;
    expect(uf::parse_request(req("Host: x\r\nContent-Length: 3\r\n", "abc"), l).status == 413,
           "body limit");
    l.header_count = 1;
    expect(uf::parse_request(req("Host: x\r\nX: y\r\n"), l).status == 431, "header count");
    l.header_bytes = 10;
    expect(uf::parse_request(base, l).status == 431, "header bytes");
    expect(uf::parse_request(req("Host: x\r\nConnection: keep-alive, Close\r\n")).request.close,
           "close token");
    char temp[] = "/tmp/unfaltering-unit-XXXXXX";
    auto dir = mkdtemp(temp);
    if (!dir)
      throw std::runtime_error("mkdtemp");
    std::filesystem::path root(dir);
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() {
        std::filesystem::remove_all(path);
      }
    } cleanup{root};
    std::ofstream(root / "index.html") << "hello";
    std::filesystem::create_directory(root / "sub");
    std::ofstream(root / "sub/index.html") << "sub";
    std::filesystem::create_symlink("/etc/passwd", root / "escape");
    std::filesystem::create_directory_symlink("/etc", root / "outside");
    auto fd = uf::open_root(root);
    expect(uf::open_resource(fd.get(), "/").length == 5, "root index");
    expect(uf::open_resource(fd.get(), "/sub/").length == 3, "nested index");
    for (auto path : {"/../passwd", "/%2e%2e/passwd", "/%252e%252e/passwd", "/escape",
                      "/outside/passwd", "/%00", "/a\\b", "/./index.html"})
      expect(uf::open_resource(fd.get(), path).status == 403, "path isolation");
    expect(uf::open_resource(fd.get(), "/%zz").status == 400, "malformed encoding");
    expect(uf::open_resource(fd.get(), "/missing").status == 404, "not found");
    auto config = root / "server.ufc";
    auto write = [&](const std::string &s) {
      std::ofstream f(config);
      f << s;
    };
    write("server { listen \"127.0.0.1:8080\"; root \".\"; host \"one.test\" { root \"sub\"; } }");
    auto c = uf::read_config(config);
    expect(c.port == 8080 && c.hosts.size() == 1, "valid config");
    for (auto text : {"server { root \".\"; }", "server { listen \"localhost:8080\"; root \".\"; }",
                      "server { listen \"127.0.0.1:8080\"; root \".\"; surprise 1; }",
                      "server { listen \"127.0.0.1:8080\"; root \".\"; root \".\"; }",
                      "server { listen \"127.0.0.1:8080\"; root \".\"; max_connections 0; }",
                      "server { listen \"127.0.0.1:8080\"; root \".\"; } junk"}) {
      write(text);
      bool failed = false;
      try {
        uf::read_config(config);
      } catch (const std::exception &) {
        failed = true;
      }
      expect(failed, "invalid config");
    }
    std::vector<uf::Route> routes{
        {false, "/assets", root}, {false, "/assets/deep", root}, {true, "/assets/special", root}};
    expect(uf::match_route(routes, "/assets/special") == &routes[2], "exact priority");
    expect(uf::match_route(routes, "/assets/deep/file") == &routes[1], "longest prefix");
    expect(uf::match_route(routes, "/assetsXYZ") == nullptr, "segment boundary");
    std::cout << checks << " unit assertions passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
