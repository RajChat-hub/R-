// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/proxy.hpp"
#include <charconv>
#include <set>
namespace uf {
namespace {
std::set<std::string> hop_headers(const MessageHead& h) {
  std::set<std::string> names={"connection","keep-alive","proxy-authenticate","proxy-authorization","te","trailer","transfer-encoding","upgrade","proxy-connection"};
  for(const auto& [key,value]:h.headers) {
    if(key!="connection") continue;
    std::string_view rest=value;
    for(;;) {
      auto comma=rest.find(',');
      auto item=rest.substr(0,comma);
      while(!item.empty()&&(item.front()==' '||item.front()=='\t')) item.remove_prefix(1);
      while(!item.empty()&&(item.back()==' '||item.back()=='\t')) item.remove_suffix(1);
      names.insert(lower(item));
      if(comma==rest.npos) break;
      rest.remove_prefix(comma+1);
    }
  }
  return names;
}
}
std::string upstream_request(const MessageHead& h,const Upstream& u,std::string_view peer,bool secure) {
  auto hop=hop_headers(h);
  std::string out=h.method+" "+h.target+" HTTP/1.1\r\nHost: "+u.host+"\r\nConnection: close\r\n";
  for(const auto& [name,value]:h.headers) {
    if(hop.contains(name)||name=="host"||name=="content-length"||name=="forwarded"||name=="x-real-ip"||name.starts_with("x-forwarded-")) continue;
    out+=name+": "+value+"\r\n";
  }
  if(u.forwarding) out+="X-Forwarded-For: "+std::string(peer)+"\r\nX-Forwarded-Proto: "+std::string(secure?"https":"http")+"\r\nX-Forwarded-Host: "+h.authority+"\r\n";
  if(h.framing==Framing::chunked) out+="Transfer-Encoding: chunked\r\n";
  else out+="Content-Length: "+std::to_string(h.framing==Framing::fixed?h.length:0)+"\r\n";
  return out+"\r\n";
}
std::string downstream_response(const MessageHead& h,bool head,bool close) {
  auto hop=hop_headers(h);
  bool length=false;
  std::string out="HTTP/1.1 "+std::to_string(h.status)+" "+h.phrase+"\r\n";
  for(const auto& [name,value]:h.headers) {
    if(name=="content-length") {length=true;continue;}
    if(hop.contains(name)) continue;
    out+=name+": "+value+"\r\n";
  }
  bool empty=head||h.status==204||h.status==304;
  if(empty) {if(length&&h.status!=204) out+="Content-Length: "+std::to_string(h.length)+"\r\n";}
  else if(h.framing==Framing::fixed) out+="Content-Length: "+std::to_string(h.length)+"\r\n";
  else out+="Transfer-Encoding: chunked\r\n";
  out+="Connection: "+std::string(close?"close":"keep-alive")+"\r\n\r\n";
  return out;
}
std::string chunk_frame(std::string_view bytes) {
  char size[32];
  auto [end,error]=std::to_chars(size,size+sizeof size,bytes.size(),16);
  (void)error;
  return std::string(size,end)+"\r\n"+std::string(bytes)+"\r\n";
}
}
