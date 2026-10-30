// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/reactor.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/socket.h>
namespace uf {
namespace {
std::size_t room(std::string& bytes,std::size_t& sent,std::size_t cap) {
  if(sent) {bytes.erase(0,sent);sent=0;}
  return bytes.size()<cap?cap-bytes.size():0;
}
void append_output(Connection& c,std::string_view bytes,const Config& cfg) {
  if(bytes.size()>room(c.output,c.sent,cfg.buffer_bytes)) throw std::runtime_error("downstream queue budget violated");
  if(c.output.empty()) c.write_deadline=Clock::now()+ms(cfg.write_ms);
  c.output.append(bytes);
}
}
void Reactor::check_proxy_budget(Connection& c) {
  if(!c.proxy) return;
  auto& p=*c.proxy;
  if(c.input.size()>cfg.buffer_bytes||c.output.size()>cfg.buffer_bytes||p.incoming.size()>cfg.buffer_bytes||p.outgoing.size()>cfg.buffer_bytes)
    throw std::runtime_error("proxy buffer invariant");
  peak_proxy_bytes=std::max(peak_proxy_bytes,c.input.size()+c.output.size()+p.incoming.size()+p.outgoing.size());
}
void Reactor::release_upstream(Connection& c) {
  if(!c.proxy) return;
  auto& p=*c.proxy;
  if(p.tracked) {if(!p.backend||!p.backend->active)throw std::runtime_error("backend active invariant");--p.backend->active;p.tracked=false;}
  if(p.registered&&p.socket) epoll_ctl(epoll.get(),EPOLL_CTL_DEL,p.socket.get(),nullptr);
  p.registered=false;
  upstream_events.erase(p.event);
  if(p.tls&&p.tls->ready()&&!p.transport_error) (void)p.tls->shutdown(); // Best-effort close_notify; never pool this socket.
  p.tls.reset();
  p.socket.reset();
}
void Reactor::proxy_failure(Connection& c,int status) {
  if(c.proxy)log.emit("{\"event\":\"proxy_failure\",\"status\":"+std::to_string(status)+
    ",\"decoded\":"+std::to_string(c.proxy->response.decoded_bytes())+
    ",\"queued\":"+std::to_string(c.proxy->incoming.size())+"}\n");
  release_upstream(c);
  c.proxy.reset();
  c.close=true;
  if(c.committed||(c.tls&&c.tls->write_pending())) {c.dead=true;return;}
  c.input.clear();c.output.clear();c.sent=0;
  prepare(c,status,c.parser.head().method=="HEAD");
}
void Reactor::start_proxy(Connection& c,std::uint64_t id,const Upstream& u) {
  if(c.parser.head().method=="CONNECT") {c.close=true;prepare(c,405);return;}
  ++c.requests;++requests;++c.epoch;
  c.writing=true;c.committed=false;
  c.close=c.parser.head().close||c.eof||stopping||c.requests>=cfg.max_requests;
  c.upload_deadline=c.deadline;
  c.write_deadline=Clock::now()+ms(cfg.write_ms);
  c.proxy=std::make_unique<ProxyTransaction>(cfg,u,c.parser.head().method=="HEAD");
  auto& p=*c.proxy;
  p.backend=&balancer.state(u);++p.backend->active;p.tracked=true;
  p.event=next_id++;
  p.connect_deadline=Clock::now()+ms(u.connect_ms);
  p.response_deadline=Clock::now()+ms(u.response_ms);
  p.send_deadline=Clock::now()+ms(u.write_ms);
  p.outgoing=upstream_request(c.parser.head(),u,c.peer,c.tls!=nullptr);
  if(p.outgoing.size()>cfg.buffer_bytes) {proxy_failure(c,431);return;}
  p.socket.reset(socket(AF_INET,SOCK_STREAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0));
  if(!p.socket) {proxy_failure(c,503);return;}
  sockaddr_in addr{};addr.sin_family=AF_INET;addr.sin_port=htons(u.port);
  inet_pton(AF_INET,u.address.c_str(),&addr.sin_addr);
  int rc=connect(p.socket.get(),reinterpret_cast<sockaddr*>(&addr),sizeof addr);
  if(rc<0&&errno!=EINPROGRESS) {proxy_failure(c,502);return;}
  p.connecting=rc<0;
  if(u.tls) {
    SSL_CTX* ctx=nullptr;
    for(const auto& [name,target]:cfg.upstreams)if(&target==&u)ctx=tls_upstreams.at(name).get();
    p.tls=std::make_unique<Tls>(ctx,p.socket.get(),&u);
  }
  upstream_events.emplace(p.event,id);
  request_pump(c);
  if(c.proxy&&!sync_upstream(c,id)) proxy_failure(c,502);
}
bool Reactor::sync_upstream(Connection& c,std::uint64_t id) {
  (void)id;
  if(!c.proxy||!c.proxy->socket) return true;
  auto& p=*c.proxy;
  std::uint32_t interest=0;
  if(p.connecting) interest=EPOLLOUT;
  else {
    if(p.sent<p.outgoing.size()&&!p.early) interest|=EPOLLOUT;
    if(!p.finished&&p.incoming.size()<cfg.buffer_bytes) interest|=EPOLLIN|EPOLLRDHUP;
  }
  if(!p.connecting&&p.tls) interest=p.tls->interest(interest);
  if(!interest) {
    if(p.registered) epoll_ctl(epoll.get(),EPOLL_CTL_DEL,p.socket.get(),nullptr);
    p.registered=false;
    return true;
  }
  epoll_event event{};event.events=interest;event.data.u64=p.event;
  if(epoll_ctl(epoll.get(),p.registered?EPOLL_CTL_MOD:EPOLL_CTL_ADD,p.socket.get(),&event)<0) return false;
  p.registered=true;
  return true;
}
void Reactor::request_pump(Connection& c) {
  if(!c.proxy||c.dead) return;
  auto& p=*c.proxy;
  if(p.early) return;
  for(;;) {
    auto available=room(p.outgoing,p.sent,cfg.buffer_bytes);
    if(available<=32) break;
    if(c.parser.done()) {
      if(!p.request_end) {
        if(c.parser.head().framing==Framing::chunked) p.outgoing+="0\r\n\r\n";
        p.request_end=true;
      }
      break;
    }
    if(c.input.empty()) break;
    std::string decoded;
    auto n=c.parser.consume(c.input,decoded,std::min<std::size_t>(16384,available-32));
    c.input.erase(0,n);
    if(c.parser.failed()) {proxy_failure(c,c.parser.error_status());return;}
    if(!decoded.empty()) p.outgoing+=c.parser.head().framing==Framing::chunked?chunk_frame(decoded):decoded;
    if(!n) break;
  }
  if(p.request_end&&p.sent==p.outgoing.size()) p.request_sent=true;
}
void Reactor::response_pump(Connection& c) {
  if(!c.proxy||c.dead) return;
  auto& p=*c.proxy;
  for(;;) {
    if(p.eof&&p.incoming.empty()&&!p.response.done()) {
      if(p.transport_error) {proxy_failure(c,502);return;}
      p.response.eof();
    }
    if(p.response.failed()) {proxy_failure(c,502);return;}
    if(!p.headers) {
      if(!p.response.ready()) {
        if(p.incoming.empty()) return;
        std::string ignored;
        auto n=p.response.consume(p.incoming,ignored,0);
        p.incoming.erase(0,n);
        if(p.response.failed()) {proxy_failure(c,502);return;}
        if(!p.response.ready()) return;
      }
      if(p.response.head().status<200) {
        if(++p.informational>8) {proxy_failure(c,502);return;}
        p.response=StreamParser(Limits{cfg.limits.header_bytes,cfg.limits.header_count,cfg.response_body_bytes},true,c.parser.head().method=="HEAD");
        continue;
      }
      if(!p.request_sent) {p.early=true;c.close=true;p.outgoing.clear();p.sent=0;}
      auto headers=downstream_response(p.response.head(),c.parser.head().method=="HEAD",c.close);
      if(headers.size()>room(c.output,c.sent,cfg.buffer_bytes)) return;
      append_output(c,headers,cfg);
      p.headers=true;
      log.emit("{\"event\":\"proxy_response\",\"status\":"+std::to_string(p.response.head().status)+"}\n");
    }
    bool chunked=p.response.head().framing==Framing::chunked||p.response.head().framing==Framing::eof;
    auto available=room(c.output,c.sent,cfg.buffer_bytes);
    if(p.response.done()) {
      if(!p.incoming.empty()) {proxy_failure(c,502);return;}
      p.finished=true;
      if(!p.terminator) {
        if(chunked) {
          if(available<5) return;
          append_output(c,"0\r\n\r\n",cfg);
        }
        p.terminator=true;
      }
      release_upstream(c);
      return;
    }
    if(p.incoming.empty()||available<=(chunked?32U:0U)) return;
    std::string decoded;
    auto n=p.response.consume(p.incoming,decoded,std::min<std::size_t>(16384,available-(chunked?32U:0U)));
    p.incoming.erase(0,n);
    if(p.response.failed()) {proxy_failure(c,502);return;}
    if(!decoded.empty()) append_output(c,chunked?chunk_frame(decoded):decoded,cfg);
    if(!n) return;
  }
}
void Reactor::upstream_write(Connection& c) {
  std::size_t budget=65536;
  while(c.proxy&&!c.dead&&budget) {
    request_pump(c);
    if(!c.proxy) return;
    auto& p=*c.proxy;
    if(p.early||p.sent==p.outgoing.size()) return;
    auto length=std::min(budget,p.outgoing.size()-p.sent);
    auto n=p.tls?p.tls->write(p.outgoing.data()+p.sent,length):send(p.socket.get(),p.outgoing.data()+p.sent,length,MSG_NOSIGNAL);
    if(n>0) {auto size=static_cast<std::size_t>(n);p.sent+=size;budget-=size;request_pump(c);}
    else if(n<0&&errno==EINTR) continue;
    else if(n<0&&(errno==EAGAIN||errno==EWOULDBLOCK)) return;
    else {proxy_failure(c,502);return;}
  }
}
void Reactor::upstream_read(Connection& c) {
  std::array<char,8192> bytes{};
  std::size_t budget=65536;
  while(c.proxy&&!c.dead&&budget) {
    response_pump(c);
    if(!c.proxy||c.proxy->finished) return;
    auto& p=*c.proxy;
    auto available=cfg.buffer_bytes-p.incoming.size();
    if(!available) return;
    auto length=std::min({bytes.size(),available,budget});
    auto n=p.tls?p.tls->read(bytes.data(),length):recv(p.socket.get(),bytes.data(),length,0);
    if(n>0) {auto size=static_cast<std::size_t>(n);p.incoming.append(bytes.data(),size);budget-=size;response_pump(c);}
    else if(n==0) {p.eof=true;release_upstream(c);response_pump(c);return;}
    else if(errno==EINTR) continue;
    else if(errno==EAGAIN||errno==EWOULDBLOCK) return;
    else {
      if(p.tls)tls_diagnostic(*p.tls,true);
      // A terminal transport error must not discard already authenticated bytes
      // retained behind downstream backpressure. Drain them through the framing
      // parser first. A fixed/chunked exact boundary can complete; an incomplete
      // or close-delimited message cannot turn this error into successful EOF.
      p.transport_error=true;p.eof=true;release_upstream(c);response_pump(c);return;
    }
  }
}
void Reactor::upstream_event(Connection& c,std::uint32_t events) {
  if(!c.proxy||!c.proxy->socket) return;
  auto& p=*c.proxy;
  if(p.connecting) {
    int error=0;
    socklen_t length=sizeof error;
    if(getsockopt(p.socket.get(),SOL_SOCKET,SO_ERROR,&error,&length)<0||error) {proxy_failure(c,502);return;}
    p.connecting=false;
  }
  if(p.tls&&!p.tls->ready()) {
    auto result=p.tls->handshake();
    if(result<0) {tls_diagnostic(*p.tls,true);proxy_failure(c,502);return;}
    if(result==0)return;
    events|=EPOLLOUT; // Start sending on the successful handshake transition.
  }
  if(events&EPOLLERR) {proxy_failure(c,502);return;}
  if(p.tls?p.tls->readable(events):(events&(EPOLLIN|EPOLLHUP|EPOLLRDHUP))) upstream_read(c);
  if(c.proxy&&c.proxy->socket&&(c.proxy->tls?c.proxy->tls->writable(events):(events&EPOLLOUT))) upstream_write(c);
}
bool Reactor::finish_proxy_response(Connection& c,std::uint64_t id) {
  release_upstream(c);
  c.proxy.reset();c.writing=false;c.committed=false;
  c.output.clear();c.sent=0;
  if(c.close||stopping) {c.graceful=true;return false;}
  c.parser=StreamParser(cfg.limits);c.started=!c.input.empty();
  c.deadline=Clock::now()+ms(c.started?cfg.request_ms:cfg.idle_ms);
  if(c.started) dispatch(c,id);
  return !c.dead;
}
bool Reactor::proxy_write_client(Connection& c,std::uint64_t id) {
  std::size_t budget=65536;
  while(c.proxy&&!c.dead&&budget) {
    response_pump(c);
    if(!c.proxy) break;
    if(c.sent==c.output.size()) {
      c.output.clear();c.sent=0;
      if(c.proxy->finished&&c.proxy->terminator) return finish_proxy_response(c,id);
      return true;
    }
    auto length=std::min(budget,c.output.size()-c.sent);
    auto n=c.tls?c.tls->write(c.output.data()+c.sent,length):send(c.socket.get(),c.output.data()+c.sent,length,MSG_NOSIGNAL);
    if(n>0) {auto size=static_cast<std::size_t>(n);c.sent+=size;budget-=size;c.committed=true;c.write_deadline=Clock::now()+ms(cfg.write_ms);}
    else if(n<0&&errno==EINTR) continue;
    else if(n<0&&(errno==EAGAIN||errno==EWOULDBLOCK)) return true;
    else return false;
  }
  // When the write budget empties a full client queue, explicitly refill it.
  // Otherwise a full upstream queue could leave neither endpoint interested.
  if(c.proxy&&!c.dead) response_pump(c);
  if(c.proxy&&!c.dead&&c.sent==c.output.size()&&c.proxy->finished&&c.proxy->terminator)
    return finish_proxy_response(c,id);
  return !c.dead;
}
}
