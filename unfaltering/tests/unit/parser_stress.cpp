// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/config.hpp"
#include "unfaltering/http.hpp"
#include <array>
#include <iostream>
#include <random>
#include <stdexcept>
int main() {
  std::mt19937 rng(0x5546U);
  const std::array<std::string, 4> seeds = {
      "GET / HTTP/1.1\r\nHost: x\r\n\r\n",
      "POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\n\r\nabc",
      "GET / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx\r\n0\r\n\r\n",
      "server { listen \"127.0.0.1:8080\"; root \"/tmp\"; }"};
  for (unsigned i = 0; i < 20000; ++i) {
    auto value = seeds[i % seeds.size()];
    for (unsigned j = 0; j < 1 + rng() % 8; ++j) {
      auto at = static_cast<std::size_t>(rng()) % (value.size() + 1);
      if (at == value.size())
        value.push_back(static_cast<char>(rng() % 256));
      else
        value[at] = static_cast<char>(rng() % 256);
    }
    if (i % 3 == 0)
      value.resize(static_cast<std::size_t>(rng()) % (value.size() + 1));
    auto parsed = uf::parse_request(value, {4096, 32, 4096});
    if (parsed.state == uf::ParseResult::State::complete &&
        (parsed.consumed > value.size() || parsed.request.body.size() > 4096))
      throw std::runtime_error("HTTP parser invariant");
    try {
      uf::parse_config(value, "/tmp/stress.ufc");
    } catch (const std::exception &) {
    }
  }
  std::cout << "20000 seeded mutations passed HTTP/config invariant checks; not coverage-guided "
               "fuzz certification\n";
}
