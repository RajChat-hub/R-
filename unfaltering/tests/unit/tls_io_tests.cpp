// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/tls.hpp"
#include "unfaltering/fd.hpp"
#include <sys/socket.h>
#include <stdexcept>
#include <iostream>
using namespace uf;
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  auto check=[](bool ok) {if(!ok)throw std::runtime_error("TLS partial IO invariant");};
  Config cfg;cfg.tls.enabled=true;cfg.tls.certificate=std::string(argv[1])+"/server.crt";cfg.tls.private_key=std::string(argv[1])+"/server.key";
  Upstream u;u.tls=true;u.verify_name="app.test";u.sni="app.test";u.ca_file=std::string(argv[1])+"/ca.crt";
  auto server_ctx=listener_context(cfg);auto client_ctx=upstream_context(u);
  int sockets[2];check(socketpair(AF_UNIX,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0,sockets)==0);
  Fd a(sockets[0]),b(sockets[1]);int small=4096;
  setsockopt(a.get(),SOL_SOCKET,SO_SNDBUF,&small,sizeof small);setsockopt(b.get(),SOL_SOCKET,SO_SNDBUF,&small,sizeof small);
  Tls server(server_ctx.get(),a.get()),client(client_ctx.get(),b.get(),&u);
  for(unsigned i=0;!server.ready()||!client.ready();++i) {
    check(i<10000);if(!server.ready())check(server.handshake()>=0);if(!client.ready())check(client.handshake()>=0);
  }
  unsigned blocked=0;
  for(unsigned direction=0;direction<2;++direction) {
    auto& writer=direction?server:client;auto& reader=direction?client:server;
    std::string source(2*1024*1024,'x');for(std::size_t i=0;i<source.size();++i)source[i]=static_cast<char>('a'+i%26);
    std::size_t sent=0,received=0;char bytes[997];
    for(unsigned i=0;received<source.size();++i) {
      check(i<100000);
      // Force WANT_WRITE before letting the peer read. Retry the stable internal
      // snapshot even as caller string addresses move between invocations.
      for(unsigned j=0;j<10&&sent<source.size();++j) {
        std::string moving=source.substr(sent,std::min<std::size_t>(16384,source.size()-sent));
        auto n=writer.write(moving.data(),moving.size());
        if(n>0)sent+=static_cast<std::size_t>(n);
        else {check(errno==EAGAIN);++blocked;break;}
      }
      auto n=reader.read(bytes,sizeof bytes);
      if(n>0) {auto count=static_cast<std::size_t>(n);check(source.substr(received,count)==std::string(bytes,count));received+=count;}
      else check(n<0&&errno==EAGAIN);
    }
    check(sent==source.size());
  }
  check(blocked>0);
  bool sc=false,cc=false;
  for(unsigned i=0;!sc||!cc;++i) {check(i<10000);if(!sc){auto n=server.shutdown();check(n>=0);sc=n==1;}if(!cc){auto n=client.shutdown();check(n>=0);cc=n==1;}}
  std::cout<<"TLS socketpair: verified handshake, 4 MiB exact payload, "<<blocked<<" WANT retries, mutual close_notify passed\n";
}
