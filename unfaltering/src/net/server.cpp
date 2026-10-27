// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/server.hpp"
#include "unfaltering/fd.hpp"
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
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <unordered_map>
#include <vector>
namespace uf {
namespace {
using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
auto ms(unsigned value) {
  return std::chrono::milliseconds(value);
}
struct Connection {
  Fd socket, file;
  std::string input, output;
  std::size_t sent = 0, file_sent = 0, file_length = 0;
  Time deadline = Clock::now();
  bool started = false, writing = false, close = false, eof = false;
  unsigned requests = 0;
  explicit Connection(int fd, unsigned idle) : socket(fd), deadline(Clock::now() + ms(idle)) {}
};
class Reactor {
  const Config &cfg;
  Fd epoll, listener, signals, reserve;
  std::map<std::string, Fd> roots;
  std::vector<Fd> route_roots;
  std::unordered_map<std::uint64_t, Connection> clients;
  std::uint64_t next_id = 3, requests = 0;
  bool stopping = false, listener_paused = false;
  Time stop_at{}, listener_retry{};
  void add(int fd, std::uint64_t id, std::uint32_t events) {
    epoll_event e{};
    e.events = events;
    e.data.u64 = id;
    if (epoll_ctl(epoll.get(), EPOLL_CTL_ADD, fd, &e) < 0)
      throw std::runtime_error("epoll add failed");
  }
  bool arm(Connection &c, std::uint64_t id) {
    epoll_event e{};
    // A client may shutdown its write half while still reading the response.
    // RDHUP is relevant while reading only; monitoring it while blocked on
    // output would repeatedly wake the reactor or prematurely truncate output.
    e.events = c.writing ? static_cast<std::uint32_t>(EPOLLOUT)
                         : static_cast<std::uint32_t>(EPOLLIN | EPOLLRDHUP);
    e.data.u64 = id;
    return epoll_ctl(epoll.get(), EPOLL_CTL_MOD, c.socket.get(), &e) == 0;
  }
  void erase(std::uint64_t id) {
    auto it = clients.find(id);
    if (it != clients.end()) {
      epoll_ctl(epoll.get(), EPOLL_CTL_DEL, it->second.socket.get(), nullptr);
      clients.erase(it);
    }
  }
  void prepare(Connection &c, int status, bool head = false) {
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
  void dispatch(Connection &c) {
    auto parsed = parse_request(c.input, cfg.limits);
    if (parsed.state == ParseResult::State::incomplete) {
      if (c.eof) {
        c.close = true;
        prepare(c, 400);
      }
      return;
    }
    if (parsed.state == ParseResult::State::error) {
      c.close = true;
      prepare(c, parsed.status, parsed.request.method == "HEAD");
      return;
    }
    auto &req = parsed.request;
    c.input.erase(0, parsed.consumed);
    ++c.requests;
    ++requests;
    c.close = req.close || c.eof || stopping || c.requests >= cfg.max_requests;
    bool head = req.method == "HEAD";
    int status = 200;
    if (req.method != "GET" && !head) {
      status = 405;
      prepare(c, status);
    } else {
      auto it = roots.find(req.host);
      if (it == roots.end())
        it = roots.find("");
      int root = it->second.get();
      if (auto route = match_route(cfg.routes, req.target)) {
        auto index = static_cast<std::size_t>(route - cfg.routes.data());
        root = route_roots[index].get();
      }
      auto resource = open_resource(root, req.target);
      status = resource.status;
      if (status != 200)
        prepare(c, status, head);
      else {
        c.output = response_headers(200, resource.length, resource.mime, c.close);
        c.sent = 0;
        c.file_sent = 0;
        c.file_length = head ? 0 : resource.length;
        c.file = std::move(resource.file);
        c.writing = true;
        c.deadline = Clock::now() + ms(cfg.write_ms);
      }
    }
    // Deliberately omit paths, query strings, headers and bodies from access logs.
    std::cerr << "{\"event\":\"request\",\"id\":" << requests << ",\"status\":" << status << "}\n";
  }
  bool read(Connection &c) {
    std::array<char, 8192> buf{};
    std::size_t budget = 65536;
    while (budget > 0) {
      auto n = recv(c.socket.get(), buf.data(), std::min(buf.size(), budget), 0);
      if (n > 0) {
        if (!c.started) {
          c.started = true;
          c.deadline = Clock::now() + ms(cfg.request_ms);
        }
        auto size = static_cast<std::size_t>(n);
        budget -= size;
        auto cap = cfg.limits.header_bytes + cfg.limits.body_bytes + 65536;
        if (c.input.size() + size > cap) {
          c.close = true;
          prepare(c, 413);
          return true;
        }
        c.input.append(buf.data(), size);
        dispatch(c);
        if (c.writing)
          return true;
      } else if (n == 0) {
        c.eof = true;
        if (c.input.empty())
          return false;
        dispatch(c);
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
  bool write(Connection &c) {
    std::size_t budget = 65536;
    while (budget > 0) {
      if (c.sent == c.output.size()) {
        if (c.file_sent < c.file_length) {
          std::array<char, 16384> buf{};
          auto want = std::min(buf.size(), c.file_length - c.file_sent);
          ssize_t n;
          do {
            n = pread(c.file.get(), buf.data(), want, static_cast<off_t>(c.file_sent));
          } while (n < 0 && errno == EINTR);
          if (n <= 0)
            return false;
          c.output.assign(buf.data(), static_cast<std::size_t>(n));
          c.file_sent += static_cast<std::size_t>(n);
          c.sent = 0;
        } else {
          c.file.reset();
          c.output.clear();
          c.sent = 0;
          c.writing = false;
          if (c.close || stopping)
            return false;
          c.started = !c.input.empty();
          c.deadline = Clock::now() + ms(c.started ? cfg.request_ms : cfg.idle_ms);
          if (c.started)
            dispatch(c);
          if (!c.writing)
            return true;
        }
      }
      auto n = send(c.socket.get(), c.output.data() + c.sent,
                    std::min(budget, c.output.size() - c.sent), MSG_NOSIGNAL);
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
  void accept_clients() {
    for (unsigned i = 0; i < 64; ++i) {
      int fd = accept4(listener.get(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC);
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
      if (clients.size() >= cfg.max_connections) {
        ::close(fd);
        continue;
      }
      auto id = next_id++;
      auto [it, inserted] = clients.try_emplace(id, fd, cfg.idle_ms);
      (void)inserted;
      try {
        add(it->second.socket.get(), id, EPOLLIN | EPOLLRDHUP);
      } catch (...) {
        clients.erase(id);
        throw;
      }
    }
  }
  void stop() {
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

public:
  explicit Reactor(const Config &config) : cfg(config), epoll(epoll_create1(EPOLL_CLOEXEC)) {
    if (!epoll)
      throw std::runtime_error("epoll creation failed");
    roots.emplace("", open_root(cfg.root));
    for (const auto &[name, path] : cfg.hosts)
      roots.emplace(name, open_root(path));
    for (const auto &route : cfg.routes)
      route_roots.push_back(open_root(route.root));
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    if (sigprocmask(SIG_BLOCK, &mask, nullptr) < 0)
      throw std::runtime_error("signal mask failed");
    signals.reset(signalfd(-1, &mask, SFD_NONBLOCK | SFD_CLOEXEC));
    if (!signals)
      throw std::runtime_error("signalfd failed");
    add(signals.get(), 2, EPOLLIN);
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
  int run() {
    std::cerr << "{\"event\":\"listening\",\"port\":" << cfg.port << "}\n";
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
        auto it = clients.find(id);
        if (it == clients.end())
          continue;
        auto &c = it->second;
        bool keep = true;
        try {
          if (e.events & EPOLLERR)
            keep = false;
          else if (c.writing) {
            if (e.events & EPOLLOUT)
              keep = write(c);
            else if (e.events & (EPOLLHUP | EPOLLRDHUP))
              keep = false;
          } else if (e.events & (EPOLLIN | EPOLLRDHUP | EPOLLHUP))
            keep = read(c);
        } catch (const std::exception &) {
          keep = false;
          std::cerr << "{\"event\":\"connection_error\"}\n";
        }
        if (!keep || !arm(c, id))
          erase(id);
      }
      auto now = Clock::now();
      if (listener_paused && !stopping && now >= listener_retry) {
        add(listener.get(), 1, EPOLLIN);
        listener_paused = false;
      }
      std::vector<std::uint64_t> drop;
      for (auto &[id, c] : clients) {
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
    std::cerr << "{\"event\":\"stopped\",\"requests\":" << requests << "}\n";
    return 0;
  }
};
} // namespace
int serve(const Config &config) {
  return Reactor(config).run();
}
} // namespace uf
