// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/health.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/socket.h>
namespace uf {
Probe::Probe(BackendState& state,SSL_CTX* context):backend(&state) {
  const auto& u=*state.target;
  deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(u.health.timeout_ms);
  outgoing_="GET "+u.health.path+" HTTP/1.1\r\nHost: "+u.host+"\r\nConnection: close\r\n\r\n";
  socket.reset(::socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0));
  if(!socket)throw std::runtime_error("probe local socket unavailable");
  sockaddr_in address{};address.sin_family=AF_INET;address.sin_port=htons(u.port);
  inet_pton(AF_INET,u.address.c_str(),&address.sin_addr);
  auto rc=connect(socket.get(),reinterpret_cast<sockaddr*>(&address),sizeof address);
  // Immediate transport errors are consumed through SO_ERROR/recv on readiness.
  // Preserve an explicit failed state rather than treating them as local overload.
  if(rc<0&&errno!=EINPROGRESS) {connecting_=false;failed_=true;}
  else connecting_=rc<0;
  if(u.tls)tls=std::make_unique<Tls>(context,socket.get(),&u);
}
std::uint32_t Probe::interest() const {
  if(connecting_)return EPOLLOUT;
  std::uint32_t events=EPOLLIN|EPOLLRDHUP;
  if(sent_<outgoing_.size())events|=EPOLLOUT;
  return tls?tls->interest(events):events;
}
int Probe::consume() {
  while(!incoming_.empty()&&!parser_.failed()&&!parser_.done()) {
    std::string sink;auto n=parser_.consume(incoming_,sink,8192);incoming_.erase(0,n);
    if(parser_.ready()&&parser_.head().status<200) {
      if(++informational_>8)return -1;
      parser_=StreamParser(Limits{8192,32,65536},true);continue;
    }
    if(!n)break;
  }
  if(parser_.failed())return -1;
  if(parser_.done())return incoming_.empty()&&static_cast<unsigned>(parser_.head().status)==backend->target->health.expect_status?1:-1;
  return 0;
}
int Probe::advance(std::uint32_t events) {
  if(failed_)return -1;
  if(connecting_) {
    int error=0;socklen_t length=sizeof error;
    if(getsockopt(socket.get(),SOL_SOCKET,SO_ERROR,&error,&length)<0||error)return -1;
    connecting_=false;
  }
  if(events&EPOLLERR)return -1;
  if(tls&&!tls->ready()) {
    auto n=tls->handshake();if(n<0)return -1;if(!n)return 0;events|=EPOLLOUT;
  }
  if(sent_<outgoing_.size()&&(tls?tls->writable(events):(events&EPOLLOUT))) {
    auto n=tls?tls->write(outgoing_.data()+sent_,outgoing_.size()-sent_):send(socket.get(),outgoing_.data()+sent_,outgoing_.size()-sent_,MSG_NOSIGNAL);
    if(n>0)sent_+=static_cast<std::size_t>(n);
    else if(n==0||(errno!=EAGAIN&&errno!=EWOULDBLOCK&&errno!=EINTR))return -1;
  }
  if(!(tls?(tls->readable(events)||tls->pending()):(events&(EPOLLIN|EPOLLRDHUP|EPOLLHUP))))return 0;
  std::array<char,8192> bytes{};std::size_t budget=65536;
  for(unsigned operations=0;budget&&operations<32;++operations) {
    if(auto result=consume())return result;
    auto available=16384-incoming_.size();if(!available)return -1;
    auto size=std::min({budget,available,bytes.size()});
    auto n=tls?tls->read(bytes.data(),size):recv(socket.get(),bytes.data(),size,0);
    if(n>0) {auto count=static_cast<std::size_t>(n);budget-=count;incoming_.append(bytes.data(),count);}
    else if(n==0) {
      if(auto result=consume())return result;
      parser_.eof();return !parser_.failed()&&parser_.done()&&static_cast<unsigned>(parser_.head().status)==backend->target->health.expect_status?1:-1;
    } else if(errno==EAGAIN||errno==EWOULDBLOCK)return consume();
    else if(errno!=EINTR)return -1;
  }
  return consume();
}
}
