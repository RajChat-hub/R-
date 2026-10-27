// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "static.hpp"
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>
namespace uf {
struct IoJob {
  enum class Kind { open, read };
  Kind kind = Kind::open;
  std::uint64_t id = 0, epoch = 0;
  Fd file;
  std::string target;
  std::size_t offset = 0, length = 0;
};
struct IoResult {
  IoJob::Kind kind = IoJob::Kind::open;
  std::uint64_t id = 0, epoch = 0;
  Resource resource;
  std::string bytes;
  bool error = false;
};
class IoPool {
  Fd event_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<IoJob> jobs_;
  std::deque<IoResult> results_;
  std::vector<std::thread> workers_;
  std::size_t outstanding_ = 0;
  bool stop_ = false;
  void work();

public:
  static constexpr std::size_t capacity = 64, block_bytes = 16384;
  IoPool();
  ~IoPool();
  void start();
  bool submit(IoJob job);
  std::deque<IoResult> take();
  int event_fd() const {
    return event_.get();
  }
};
// Fixed-capacity, lossy operational logging. Never waits for a sink on reactor.
class EventLog {
  Fd sink_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<std::string> queue_;
  std::thread worker_;
  bool stop_ = false;
  std::size_t dropped_ = 0;
  void work();

public:
  void start();
  ~EventLog();
  void emit(std::string record);
};
} // namespace uf
