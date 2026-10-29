// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
#include "fd.hpp"
#include "tls.hpp"
#include <chrono>
#include <cstdint>
namespace uf {
struct ProxyTransaction {
  Fd socket;
  std::unique_ptr<Tls> tls;
  const Upstream* target=nullptr;
  StreamParser response;
  std::string outgoing,incoming;
  std::size_t sent=0;
  std::uint64_t event=0;
  bool connecting=true,registered=false,request_end=false,request_sent=false;
  bool headers=false,finished=false,terminator=false,early=false,eof=false,transport_error=false;
  unsigned informational=0;
  std::chrono::steady_clock::time_point connect_deadline,response_deadline,send_deadline;
  ProxyTransaction(const Config& c,const Upstream& u,bool head)
      :target(&u),response(Limits{c.limits.header_bytes,c.limits.header_count,c.response_body_bytes},true,head) {}
};
std::string upstream_request(const MessageHead& request,const Upstream& target,std::string_view peer,bool secure=false);
std::string downstream_response(const MessageHead& response,bool head,bool close);
std::string chunk_frame(std::string_view bytes);
}
