// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/server.hpp"
#include "unfaltering/connection.hpp"
#include "unfaltering/fd.hpp"
#include "unfaltering/io_pool.hpp"
#include "unfaltering/static.hpp"
#include <arpa/inet.h>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <map>
#include <signal.h>
#include <stdexcept>
#include <sys/epoll.h>
#include <sys/resource.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <unordered_map>
#include <vector>
#include "unfaltering/reactor.hpp"
namespace uf {
void Reactor::add(int fd, std::uint64_t id, std::uint32_t events) {
    epoll_event e{};
    e.events = events;
    e.data.u64 = id;
    if (epoll_ctl(epoll.get(), EPOLL_CTL_ADD, fd, &e) < 0)
      throw std::runtime_error("epoll add failed");
  }

bool Reactor::arm(Connection &c, std::uint64_t id) {
  if(c.dead) return false;
  epoll_event e{};
  if(c.tls&&(!c.tls->ready()||c.tls_closing)) {
    e.events=c.tls->interest(0);e.data.u64=id;
    return epoll_ctl(epoll.get(),EPOLL_CTL_MOD,c.socket.get(),&e)==0;
  }
  if(c.proxy) {
    check_proxy_budget(c);
    auto& p=*c.proxy;
    if(!p.early&&!c.parser.done()&&!c.eof&&c.input.size()<cfg.buffer_bytes&&p.outgoing.size()-p.sent<cfg.buffer_bytes-32) e.events|=EPOLLIN|EPOLLRDHUP;
    if(c.sent<c.output.size()) e.events|=EPOLLOUT;
    if(!sync_upstream(c,id)) return false;
  } else e.events=c.writing?(c.sent<c.output.size()?static_cast<std::uint32_t>(EPOLLOUT):0U):static_cast<std::uint32_t>(EPOLLIN|EPOLLRDHUP);
  if(c.tls)e.events=c.tls->interest(e.events);
  e.data.u64=id;
  return epoll_ctl(epoll.get(),EPOLL_CTL_MOD,c.socket.get(),&e)==0;
}

void Reactor::tls_diagnostic(const Tls& tls,bool upstream) {
  log.emit("{\"event\":\"tls_error\",\"upstream\":"+std::string(upstream?"true":"false")+
    ",\"code\":"+std::to_string(tls.error)+",\"verify\":"+std::to_string(tls.verification)+"}\n");
}
bool Reactor::tls_close(Connection& c) {
  if(!c.tls||!c.tls->ready()||!c.graceful||c.dead)return false;
  if(!c.tls_closing) {
    c.tls_closing=true;c.tls_deadline=Clock::now()+ms(cfg.tls.shutdown_ms);
    release_upstream(c);c.proxy.reset();
  }
  return c.tls->shutdown()==0;
}
void Reactor::erase(std::uint64_t id) {
    auto it = clients.find(id);
    if (it != clients.end()) {
      release_upstream(it->second);
      epoll_ctl(epoll.get(), EPOLL_CTL_DEL, it->second.socket.get(), nullptr);
      clients.erase(it);
    }
  }

void Reactor::prepare(Connection &c, int status, bool head) {
    ++c.epoch;
    c.io_pending = false;
    auto body = reason(status) + "\n";
    c.output =
        response_headers(status, body.size(), "text/plain; charset=utf-8", c.close, status == 405);
    if (!head)
      c.output += body;
    c.sent = 0;
    c.file.reset();
    c.file_sent = c.file_length = 0;
    c.writing = true;
    c.deadline = Clock::now() + ms(cfg.write_ms);
  }

void Reactor::dispatch(Connection &c, std::uint64_t id) {
  if(c.proxy) {request_pump(c);return;}
  std::string discard;
  while(!c.parser.ready()&&!c.parser.failed()&&!c.input.empty()) {
    auto n=c.parser.consume(c.input,discard,16384);
    c.input.erase(0,n);
    if(!n) break;
  }
  if(c.parser.ready()&&!c.parser.failed()) {
    if(auto route=match_route(cfg.routes,c.parser.head().target);route&&!route->upstream.empty()) {
      start_proxy(c,id,cfg.upstreams.at(route->upstream));
      return;
    }
  }
  while(!c.parser.done()&&!c.parser.failed()&&!c.input.empty()) {
    discard.clear();
    auto n=c.parser.consume(c.input,discard,16384);
    c.input.erase(0,n);
    if(!n) break;
  }
  if(c.parser.failed()) {c.close=true;prepare(c,c.parser.error_status(),c.parser.head().method=="HEAD");return;}
  if(!c.parser.done()) {if(c.eof) {c.close=true;prepare(c,400);}return;}
    const auto &req = c.parser.head();
    ++c.requests;
    ++requests;
    ++c.epoch;
    c.close = req.close || c.eof || stopping || c.requests >= cfg.max_requests;
    c.writing = true;
    c.deadline = Clock::now() + ms(cfg.write_ms);
    if (req.method != "GET" && req.method != "HEAD") {
      prepare(c, 405);
      return;
    }
    auto it = roots.find(req.host);
    if (it == roots.end())
      it = roots.find("");
    int root = it->second.get();
    if (auto route = match_route(cfg.routes, req.target))
      root = route_roots[static_cast<std::size_t>(route - cfg.routes.data())].get();
    IoJob job;
    job.id = id;
    job.epoch = c.epoch;
    job.file.reset(fcntl(root, F_DUPFD_CLOEXEC, 0));
    job.target = req.target;
    if (!job.file || !io.submit(std::move(job))) {
      c.close = true;
      prepare(c, 503);
      return;
    }
    c.io_pending = true;
  }

bool Reactor::read_file(Connection &c, std::uint64_t id) {
    if (c.io_pending)
      return true;
    IoJob job;
    job.kind = IoJob::Kind::read;
    job.id = id;
    job.epoch = c.epoch;
    job.offset = c.file_sent;
    job.length = std::min(IoPool::block_bytes, c.file_length - c.file_sent);
    job.file.reset(fcntl(c.file.get(), F_DUPFD_CLOEXEC, 0));
    if (!job.file)
      return false;
    if (io.submit(std::move(job)))
      c.io_pending = true;
    return true;
  }

void Reactor::completed_io() {
    for (auto &result : io.take()) {
      auto it = clients.find(result.id);
      if (it == clients.end() || it->second.epoch != result.epoch)
        continue;
      auto &c = it->second;
      c.io_pending = false;
      if (result.error) {
        erase(result.id);
        continue;
      }
      if (result.kind == IoJob::Kind::open) {
        auto &resource = result.resource;
        if (resource.status != 200)
          prepare(c, resource.status, c.parser.head().method == "HEAD");
        else {
          c.output = response_headers(200, resource.length, resource.mime, c.close);
          c.sent = 0;
          c.file_sent = 0;
          c.file_length = c.parser.head().method == "HEAD" ? 0 : resource.length;
          c.file = std::move(resource.file);
        }
        log.emit("{\"event\":\"request\",\"status\":" + std::to_string(resource.status) + "}\n");
      } else {
        c.output = std::move(result.bytes);
        c.sent = 0;
        c.file_sent += c.output.size();
      }
      if (!arm(c, result.id))
        erase(result.id);
    }
  }

bool Reactor::read(Connection &c, std::uint64_t id) {
    std::array<char, 8192> buf{};
    std::size_t budget = 65536;
    while (budget > 0) {
      auto available=cfg.buffer_bytes-c.input.size();
      if(!available) return true;
      auto length=std::min({buf.size(),budget,available});
      auto n = c.tls?c.tls->read(buf.data(),length):recv(c.socket.get(),buf.data(),length,0);
      if (n > 0) {
        if (!c.started) {
          c.started = true;
          c.deadline = Clock::now() + ms(cfg.request_ms);
        }
        auto size = static_cast<std::size_t>(n);
        budget -= size;
        auto cap = cfg.buffer_bytes;
        if (c.input.size() + size > cap) {
          c.close = true;
          prepare(c, 413);
          return true;
        }
        c.input.append(buf.data(), size);
        dispatch(c, id);
        if(c.dead) return false;
        if (c.writing)
          return true;
      } else if (n == 0) {
        c.eof = true;
        if(c.proxy) {
          c.parser.eof();
          if(c.parser.failed()) proxy_failure(c,400);
          else request_pump(c);
          return !c.dead;
        }
        if (c.input.empty())
          return false;
        dispatch(c, id);
        return true;
      } else if (errno == EINTR)
        continue;
      else if (errno == EAGAIN || errno == EWOULDBLOCK)
        return true;
      else
        return false;
    }
    return true;
  }

bool Reactor::write(Connection &c, std::uint64_t id) {
    if(c.proxy) return proxy_write_client(c,id);
    std::size_t budget = 65536;
    while (budget > 0) {
      if (c.sent == c.output.size()) {
        if (c.file_sent < c.file_length) {
          return read_file(c, id);
        } else {
          c.file.reset();
          c.output.clear();
          c.sent = 0;
          c.writing = false;
          if (c.close || stopping) {c.graceful=true;return false;}
          c.parser = StreamParser(cfg.limits);
          c.started = !c.input.empty();
          c.deadline = Clock::now() + ms(c.started ? cfg.request_ms : cfg.idle_ms);
          if (c.started)
            dispatch(c, id);
          if (!c.writing)
            return true;
        }
      }
      if(c.sent==c.output.size())return true; // Waiting for bounded file completion.
      auto length=std::min(budget,c.output.size()-c.sent);
      auto n = c.tls?c.tls->write(c.output.data()+c.sent,length):send(c.socket.get(),c.output.data()+c.sent,length,MSG_NOSIGNAL);
      if (n > 0) {
        auto size = static_cast<std::size_t>(n);
        c.sent += size;
        budget -= size;
      } else if (n < 0 && errno == EINTR)
        continue;
      else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        return true;
      else
        return false;
    }
    return true;
  }

void Reactor::accept_clients() {
    for (unsigned i = 0; i < 64; ++i) {
      sockaddr_in peer{};
      socklen_t peer_size=sizeof peer;
      int fd=accept4(listener.get(),reinterpret_cast<sockaddr*>(&peer),&peer_size,SOCK_NONBLOCK|SOCK_CLOEXEC);
      if (fd < 0) {
        if (errno == EINTR)
          continue;
        // On descriptor exhaustion, use a reserve descriptor to consume one pending
        // accept; disable the listener briefly to prevent a hot readiness loop.
        if (errno == EMFILE || errno == ENFILE) {
          reserve.reset();
          int dropped = accept4(listener.get(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
          if (dropped >= 0)
            ::close(dropped);
          reserve.reset(::open("/dev/null", O_RDONLY | O_CLOEXEC));
          epoll_ctl(epoll.get(), EPOLL_CTL_DEL, listener.get(), nullptr);
          listener_paused = true;
          listener_retry = Clock::now() + ms(100);
        }
        return;
      }
      if (clients.size() >= cfg.max_connections ||
          (clients.size() + 1) * connection_reservation(cfg) > cfg.memory_budget_bytes) {
        ::close(fd);
        continue;
      }
      auto id = next_id++;
      auto [it, inserted] = clients.try_emplace(id, fd, cfg);
      (void)inserted;
      char peer_text[INET_ADDRSTRLEN]{};
      inet_ntop(AF_INET,&peer.sin_addr,peer_text,sizeof peer_text);
      it->second.peer=peer_text;
      try {
        if(tls_listener) {
          it->second.tls=std::make_unique<Tls>(tls_listener.get(),fd);
          it->second.tls_deadline=Clock::now()+ms(cfg.tls.handshake_ms);
        }
        add(it->second.socket.get(), id, tls_listener?static_cast<std::uint32_t>(EPOLLIN|EPOLLOUT):static_cast<std::uint32_t>(EPOLLIN|EPOLLRDHUP));
      } catch (...) {
        clients.erase(id);
        throw;
      }
    }
  }

void Reactor::stop() {
    if (stopping)
      return;
    stopping = true;
    stop_at = Clock::now() + ms(cfg.shutdown_ms);
    epoll_ctl(epoll.get(), EPOLL_CTL_DEL, listener.get(), nullptr);
    listener.reset();
    std::vector<std::uint64_t> drop;
    for (auto &[id, c] : clients) {
      c.close = true;
      if (!c.writing)
        drop.push_back(id);
    }
    for (auto id : drop)
      erase(id);
  }

Reactor::Reactor(const Config &config) : cfg(config), epoll(epoll_create1(EPOLL_CLOEXEC)) {
    if (!epoll)
      throw std::runtime_error("epoll creation failed");
    tls_listener=listener_context(cfg);
    for(const auto& [name,u]:cfg.upstreams) if(u.tls)tls_upstreams.emplace(name,upstream_context(u));
    roots.emplace("", open_root(cfg.root));
    for (const auto &[name, path] : cfg.hosts)
      roots.emplace(name, open_root(path));
    for (const auto &route : cfg.routes)
      route_roots.push_back(route.upstream.empty()?open_root(route.root):Fd());
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGPIPE);
    if (sigprocmask(SIG_BLOCK, &mask, nullptr) < 0)
      throw std::runtime_error("signal mask failed");
    sigdelset(&mask,SIGPIPE);
    signals.reset(signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC));
    if (!signals)
      throw std::runtime_error("signalfd failed");
    add(signals.get(), 2, EPOLLIN);
    io.start();
    log.start();
    add(io.event_fd(), 3, EPOLLIN);
    reserve.reset(::open("/dev/null", O_RDONLY | O_CLOEXEC));
    listener.reset(socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
    if (!listener)
      throw std::runtime_error("socket failed");
    int yes = 1;
    if (setsockopt(listener.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes) < 0)
      throw std::runtime_error("setsockopt failed");
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(cfg.port);
    if (inet_pton(AF_INET, cfg.bind.c_str(), &addr.sin_addr) != 1)
      throw std::runtime_error("invalid bind address");
    if (bind(listener.get(), reinterpret_cast<sockaddr *>(&addr), sizeof addr) < 0 ||
        listen(listener.get(), 128) < 0)
      throw std::runtime_error("bind/listen failed");
    add(listener.get(), 1, EPOLLIN);
  }

int Reactor::run() {
    log.emit("{\"event\":\"listening\",\"port\":" + std::to_string(cfg.port) + "}\n");
    std::array<epoll_event, 128> events{};
    while (!stopping || !clients.empty()) {
      int n = epoll_wait(epoll.get(), events.data(), static_cast<int>(events.size()), 50);
      if (n < 0) {
        if (errno == EINTR)
          continue;
        throw std::runtime_error("epoll wait failed");
      }
      for (int i = 0; i < n; ++i) {
        auto e = events[static_cast<std::size_t>(i)];
        auto id = e.data.u64;
        if (id == 3) {
          completed_io();
          continue;
        }
        if (id == 2) {
          signalfd_siginfo info{};
          while (::read(signals.get(), &info, sizeof info) > 0) {
          }
          stop();
          continue;
        }
        if (id == 1) {
          if (!stopping)
            accept_clients();
          continue;
        }
        if(auto upstream=upstream_events.find(id);upstream!=upstream_events.end()) {
        auto client=clients.find(upstream->second);
        if(client==clients.end()) continue;
        auto client_id=client->first;
        auto& c=client->second;
        try {upstream_event(c,e.events);} catch(const std::exception&) {proxy_failure(c,502);}
        if(c.dead||!arm(c,client_id)) erase(client_id);
        continue;
      }
      auto it = clients.find(id);
        if (it == clients.end())
          continue;
        auto &c = it->second;
        bool keep = true;
        try {
          if (e.events & EPOLLERR)
            keep = false;
          else if(c.tls&&c.tls_closing) keep=tls_close(c);
          else if(c.tls&&!c.tls->ready()) {
            int result=c.tls->handshake();
            if(result<0) {tls_diagnostic(*c.tls,false);keep=false;}
            else if(result==1) {
              c.deadline=Clock::now()+ms(cfg.idle_ms);
              keep=read(c,id);
            }
          } else if(c.proxy) {
            if(c.tls?c.tls->readable(e.events):(e.events&(EPOLLIN|EPOLLRDHUP|EPOLLHUP))) keep=read(c,id);
            if(keep&&!c.dead&&(c.tls?c.tls->writable(e.events):(e.events&EPOLLOUT))) keep=write(c,id);
          } else if (c.writing) {
            if (c.tls?c.tls->writable(e.events):(e.events & EPOLLOUT))
              keep = write(c, id);
            else if (e.events & (EPOLLHUP | EPOLLRDHUP))
              keep = false;
          } else if (c.tls?c.tls->readable(e.events):(e.events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP)))
            keep = read(c, id);
        } catch (const std::exception &) {
          keep = false;
          log.emit("{\"event\":\"connection_error\"}\n");
        }
        if(!keep&&c.graceful)keep=tls_close(c);
        if (!keep || !arm(c, id))
          erase(id);
      }
      peak_network_endpoints=std::max(peak_network_endpoints,clients.size()+upstream_events.size());
      auto now = Clock::now();
      if (listener_paused && !stopping && now >= listener_retry) {
        add(listener.get(), 1, EPOLLIN);
        listener_paused = false;
      }
      std::vector<std::uint64_t> drop;
      for (auto &[id, c] : clients) {
        if(c.tls) {
          if(c.tls_closing||!c.tls->ready()) {
            if(now>=c.tls_deadline||(stopping&&now>=stop_at))drop.push_back(id);
            continue;
          }
          // OpenSSL can retain plaintext after socket readiness has drained.
          if(c.tls->pending()&&(!c.writing||c.proxy)&&c.input.size()<cfg.buffer_bytes) {
            bool keep=read(c,id);
            if(!keep||c.dead) {drop.push_back(id);continue;}
            if(!arm(c,id)) {drop.push_back(id);continue;}
          }
        }
        if(c.proxy) {
          if(stopping&&now>=stop_at) {drop.push_back(id);continue;}
          if(c.proxy->tls&&c.proxy->socket&&c.proxy->tls->ready()&&c.proxy->tls->pending()&&c.proxy->incoming.size()<cfg.buffer_bytes) {
            upstream_read(c);
            if(!c.proxy) {if(c.dead||!arm(c,id))drop.push_back(id);continue;}
          }
          auto& p=*c.proxy;
          if((p.connecting||(p.tls&&!p.tls->ready()))&&now>=p.connect_deadline) proxy_failure(c,504);
          else if(now>=p.response_deadline) proxy_failure(c,504);
          else if(!p.early&&!p.request_sent&&now>=p.send_deadline) proxy_failure(c,504);
          else if(!p.early&&!c.parser.done()&&now>=c.upload_deadline) proxy_failure(c,408);
          else if(c.sent<c.output.size()&&now>=c.write_deadline) c.dead=true;
          if(c.dead||!arm(c,id)) drop.push_back(id);
          continue;
        }
        if (c.writing && !c.io_pending && c.sent == c.output.size() &&
            c.file_sent < c.file_length) {
          if (!read_file(c, id))
            drop.push_back(id);
        }
        if (stopping && now >= stop_at) {
          drop.push_back(id);
          continue;
        }
        if (now >= c.deadline) {
          if (c.writing || !c.started)
            drop.push_back(id);
          else {
            c.close = true;
            prepare(c, 408);
            if (!arm(c, id))
              drop.push_back(id);
          }
        }
      }
      for (auto id : drop)
        erase(id);
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    log.finish("{\"rss_kib\":"+std::to_string(usage.ru_maxrss)+",\"peak_network_endpoints\":"+std::to_string(peak_network_endpoints)+",\"event\":\"stopped\",\"requests\":"+std::to_string(requests)+",\"peak_proxy_queue_bytes\":"+std::to_string(peak_proxy_bytes)+"}\n");
    return 0;
  }
int serve(const Config& config) {return Reactor(config).run();}
}
