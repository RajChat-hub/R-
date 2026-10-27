// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/server.hpp"
#include "unfaltering/static.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc, char **argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "UNFALTERING 0.2.0-dev | SalesAutoMate | Linux epoll / HTTP/1.1 development "
                   "milestone\n";
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << "UNFALTERING Web Server\nBuilt to Endure. Engineered to Serve.\nUsage:\n  "
                   "unfaltering --version\n  unfaltering --help\n  unfaltering --check --config "
                   "FILE\n  unfaltering serve --config FILE\n";
      return 0;
    }
    if (argc != 4 || std::string(argv[2]) != "--config" ||
        (std::string(argv[1]) != "serve" && std::string(argv[1]) != "--check"))
      throw std::runtime_error("invalid arguments; use --help");
    auto config = uf::read_config(argv[3]);
    if (std::string(argv[1]) == "--check") {
      auto root = uf::open_root(config.root);
      for (const auto &[name, path] : config.hosts) {
        (void)name;
        auto fd = uf::open_root(path);
      }
      for (const auto &route : config.routes) {
        if(!route.upstream.empty()) continue;
        auto fd = uf::open_root(route.root);
      }
      std::cout << "Configuration valid; root directories opened successfully\n";
      return 0;
    }
    return uf::serve(config);
  } catch (const std::exception &e) {
    std::cerr << "UNFALTERING: " << e.what() << '\n';
    return 1;
  }
}
