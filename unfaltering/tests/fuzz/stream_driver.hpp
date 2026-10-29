// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "unfaltering/http.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
struct FuzzResult {
  bool failed,done;int status;std::size_t decoded,wire;std::uint64_t digest;
  bool operator==(const FuzzResult&) const=default;
};
inline FuzzResult drive(const std::uint8_t* data,std::size_t size,bool response,bool head,bool fragment) {
  uf::StreamParser parser({8192,64,32768},response,head);
  std::size_t at=0,iterations=0;std::string sink;std::uint64_t digest=1469598103934665603ULL;
  while(at<size&&!parser.failed()&&!parser.done()) {
    if(++iterations>2*size+16)std::abort();
    auto width=fragment?1+static_cast<std::size_t>(data[at]%127):size-at;
    auto cap=fragment?static_cast<std::size_t>(data[at]%129):4096;
    auto piece=std::string_view(reinterpret_cast<const char*>(data+at),std::min(size-at,width));
    auto n=parser.consume(piece,sink,cap);
    if(sink.size()>cap)std::abort();
    if(!n&&!parser.failed()&&!parser.done()) {
      cap=4096;n=parser.consume(piece,sink,cap);
    }
    for(unsigned char c:sink) {digest^=c;digest*=1099511628211ULL;}
    at+=n;sink.clear();if(!n)break;
  }
  parser.eof();
  if(parser.decoded_bytes()>32768||parser.wire_bytes()>size)std::abort();
  return {parser.failed(),parser.done(),parser.error_status(),parser.decoded_bytes(),parser.wire_bytes(),digest};
}
inline int fuzz_stream(const std::uint8_t* data,std::size_t size,bool response) {
  if(size>65536)return 0;
  auto compare=[&](bool head) {
    auto a=drive(data,size,response,head,false),b=drive(data,size,response,head,true);
    // Invalid bodies may expose a different valid prefix before rejection.
    // Acceptance, error category and successful boundaries must be invariant.
    if(a.failed!=b.failed||a.done!=b.done||a.status!=b.status||(!a.failed&&a!=b))std::abort();
  };
  compare(false);if(response)compare(true);
  return 0;
}
