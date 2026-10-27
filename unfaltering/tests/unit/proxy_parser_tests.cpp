// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/proxy.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>
void check(bool ok,const char* name) {if(!ok) throw std::runtime_error(name);}
int main() {
  try {
    for(const std::string& wire:std::vector<std::string>{
      "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello",
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\n\r\n"}) {
      uf::StreamParser parser({},true);std::string out,total;
      for(char c:wire) {
        check(parser.consume(std::string_view(&c,1),out,3)==1,"fragment progress");
        total+=out;out.clear();
      }
      check(parser.done()&&!parser.failed()&&total=="hello","response fragmentation");
      check(parser.wire_bytes()==wire.size(),"response work count");
    }
    for(const std::string& wire:std::vector<std::string>{
      "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nContent-Length: 1\r\n\r\nx",
      "HTTP/1.1 200 OK\r\nContent-Length: 1\r\nTransfer-Encoding: chunked\r\n\r\n",
      "HTTP/1.1 200 OK\r\nTransfer-Encoding: gzip\r\n\r\n",
      "HTTP/1.1 101 Switching Protocols\r\n\r\n"}) {
      uf::StreamParser parser({},true);std::string out;std::size_t offset=0;
      while(offset<wire.size()&&!parser.failed()&&!parser.done()) {
        auto n=parser.consume(std::string_view(wire).substr(offset),out,16);
        offset+=n;out.clear();if(!n) break;
      }
      check(parser.failed()&&parser.error_status()==502,"bad response framing");
    }
    uf::StreamParser close({},true);std::string out;
    std::string wire="HTTP/1.0 200 OK\r\n\r\nhello";
    auto n=close.consume(wire,out,16);
    close.consume(std::string_view(wire).substr(n),out,16);
    check(!close.done(),"eof not guessed");close.eof();
    check(close.done()&&out=="hello","close-delimited framing");
    uf::StreamParser head({},true,true);out.clear();
    head.consume("HTTP/1.1 200 OK\r\nContent-Length: 123\r\n\r\n",out,16);
    check(head.done()&&out.empty(),"HEAD metadata");
    uf::MessageHead request;request.method="GET";request.target="/";request.authority="public.test";
    request.headers={{"connection","X-Hop"},{"x-hop","secret"},{"forwarded","evil"},{"x-forwarded-for","evil"}};
    uf::Upstream u;u.host="app.internal";
    auto sent=uf::upstream_request(request,u,"127.0.0.1");
    check(sent.find("evil")==sent.npos&&sent.find("secret")==sent.npos,"strip nominated and forged metadata");
    check(sent.find("Host: app.internal")!=sent.npos,"controlled Host");
    std::cout<<"Response fragmentation, framing and forwarding-policy checks passed\n";
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
