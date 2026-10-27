// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/config.hpp"
#include <cstdint>
#include <exception>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
  if (size > 65536)
    return 0;
  try {
    uf::parse_config(std::string_view(reinterpret_cast<const char *>(data), size), "/tmp/fuzz.ufc");
  } catch (const std::exception &) {
  }
  return 0;
}
