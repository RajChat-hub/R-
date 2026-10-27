// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/http.hpp"
#include "unfaltering/io_pool.hpp"
#include <iostream>
#include <stdexcept>
void require(bool ok,const char* name){if(!ok)throw std::runtime_error(name);}
int main(){
 try{
  std::string payload(65536,'x');auto wire="POST / HTTP/1.1\r\nHost: x\r\nContent-Length: "+std::to_string(payload.size())+"\r\n\r\n"+payload;
  uf::StreamParser p;std::string out;std::size_t body=0;
  for(char c:wire){std::string one(1,c);auto n=p.consume(one,out,7);require(n==1,"one-byte progress");require(out.size()<=7,"sink budget");body+=out.size();out.clear();}
  require(p.done()&&body==payload.size(),"fragmented payload");require(p.wire_bytes()==wire.size(),"no wire rescanning");
  uf::StreamParser chunk;wire="POST / HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n";for(unsigned i=0;i<4000;++i)wire+="1\r\nx\r\n";wire+="0\r\n\r\n";body=0;
  for(char c:wire){chunk.consume(std::string_view(&c,1),out,7);body+=out.size();out.clear();}
  require(chunk.done()&&body==4000&&chunk.wire_bytes()==wire.size(),"tiny fragmented chunks linear");
  uf::StreamParser limited;wire="POST / HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nabcdefghij";
  auto n=limited.consume(wire,out,3);n+=limited.consume(std::string_view(wire).substr(n),out,3);
  require(out.size()==3&&!limited.done(),"full sink stops decoder");
  auto at=n;require(limited.consume(std::string_view(wire).substr(at),out,3)==0,"backpressure no read-ahead");
  out.clear();while(at<wire.size()){auto used=limited.consume(std::string_view(wire).substr(at),out,3);require(used>0,"resume progress");at+=used;out.clear();}require(limited.done(),"sink resume");
  uf::IoPool pool;auto enqueue=[&]{uf::IoJob job;return pool.submit(std::move(job));};for(std::size_t i=0;i<uf::IoPool::capacity;++i)require(enqueue(),"bounded worker admission");require(!enqueue(),"worker full rejects");
  std::cout<<"streaming fragmentation, work-count, backpressure and worker budget checks passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
