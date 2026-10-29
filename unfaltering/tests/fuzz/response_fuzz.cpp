// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "stream_driver.hpp"
extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data,std::size_t size) {return fuzz_stream(data,size,true);}
