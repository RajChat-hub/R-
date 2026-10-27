// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/http.hpp"
#include <cstdint>
#include <cstdlib>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t *data, std::size_t size) {
  uf::Limits limits{4096, 32, 4096};
  if (size > 16384)
    return 0;
  auto result =
      uf::parse_request(std::string_view(reinterpret_cast<const char *>(data), size), limits);
  if (result.state == uf::ParseResult::State::complete &&
      (result.consumed > size || result.request.body.size() > limits.body_bytes))
    std::abort();
  return 0;
}
