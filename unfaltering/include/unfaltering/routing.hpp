// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>
namespace uf {
struct Route {
  bool exact = true;
  std::string path;
  std::filesystem::path root;
};
// Exact wins, then longest prefix; prefixes match complete path segments.
const Route *match_route(const std::vector<Route> &routes, std::string_view target);
} // namespace uf
