// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "fd.hpp"
#include <filesystem>
#include <string>
#include <string_view>
namespace uf {
struct Resource {
  Fd file;
  std::size_t length = 0;
  std::string mime = "text/plain; charset=utf-8";
  int status = 404;
};
Fd open_root(const std::filesystem::path &path);
Resource open_resource(int root, std::string_view target);
} // namespace uf
