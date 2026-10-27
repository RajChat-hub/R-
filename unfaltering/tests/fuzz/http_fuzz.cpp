// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/http.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,std::size_t size) {
  if(size>16384) return 0;
  uf::StreamParser parser({4096,32,4096});std::size_t at=0;std::string sink;
  while(at<size&&!parser.failed()&&!parser.done()) {
    auto n=parser.consume(std::string_view(reinterpret_cast<const char*>(data+at),std::min<std::size_t>(size-at,13)),sink,31);
    at+=n;sink.clear();if(!n) break;
  }
  parser.eof();
  if(parser.decoded_bytes()>4096||parser.wire_bytes()>size) std::abort();
  return 0;
}
