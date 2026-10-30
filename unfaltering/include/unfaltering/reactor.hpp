// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "connection.hpp"
#include "io_pool.hpp"
#include <map>
#include <unordered_map>
namespace uf {
class Reactor {
  const Config &cfg;
  Fd epoll, signals, reserve;
  struct BoundListener {
    Listener config;Fd socket;TlsContext context;
    std::uint64_t event=0;std::size_t active=0;
    bool paused=false;Time retry{};
  };
  std::vector<BoundListener> listeners;
  std::map<std::string, Fd> roots;
  std::vector<Fd> route_roots;
  std::map<std::string,TlsContext> tls_upstreams;
  IoPool io;
  EventLog log;
  std::unordered_map<std::uint64_t, Connection> clients;
  std::uint64_t next_id = 4, requests = 0;
  bool stopping = false;
  Time stop_at{};

  std::unordered_map<std::uint64_t,std::uint64_t> upstream_events;
  std::size_t peak_proxy_bytes=0,peak_network_endpoints=0;
  void check_proxy_budget(Connection& c);
  void start_proxy(Connection& c,std::uint64_t id,const Upstream& u);
  void request_pump(Connection& c);
  void response_pump(Connection& c);
  void release_upstream(Connection& c);
  void proxy_failure(Connection& c,int status);
  bool sync_upstream(Connection& c,std::uint64_t id);
  void upstream_event(Connection& c,std::uint32_t events);
  void upstream_write(Connection& c);
  void upstream_read(Connection& c);
  bool finish_proxy_response(Connection& c,std::uint64_t id);
  bool proxy_write_client(Connection& c,std::uint64_t id);
  void add(int fd, std::uint64_t id, std::uint32_t events);
  bool arm(Connection &c, std::uint64_t id);
  void erase(std::uint64_t id);
  bool tls_close(Connection& c);
  void tls_diagnostic(const Tls& tls,bool upstream);
  void prepare(Connection &c, int status, bool head = false);
  void dispatch(Connection &c, std::uint64_t id);
  bool read_file(Connection &c, std::uint64_t id);
  void completed_io();
  bool read(Connection &c, std::uint64_t id);
  bool write(Connection &c, std::uint64_t id);
  void accept_clients(std::size_t index);
  void stop();
public:
  explicit Reactor(const Config &config);
  int run();
};
}
