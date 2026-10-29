// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/io_pool.hpp"
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdexcept>
#include <sys/eventfd.h>
namespace uf {
IoPool::IoPool() : event_(eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC)) {
  if (!event_)
    throw std::runtime_error("worker eventfd failed");
}
void IoPool::start() {
  for (unsigned i = 0; i < 2; ++i)
    workers_.emplace_back([this] { work(); });
}
void IoPool::shutdown() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
    outstanding_ -= jobs_.size();
    jobs_.clear();
  }
  cv_.notify_all();
}
IoPool::~IoPool() {
  shutdown();
  for (auto &w : workers_)
    if (w.joinable())
      w.join();
}
bool IoPool::submit(IoJob job) {
  std::lock_guard lock(mutex_);
  if (stop_ || outstanding_ >= capacity)
    return false;
  jobs_.push_back(std::move(job));
  ++outstanding_;
  cv_.notify_one();
  return true;
}
std::deque<IoResult> IoPool::take() {
  std::uint64_t n;
  while (::read(event_.get(), &n, sizeof n) > 0) {
  }
  std::lock_guard lock(mutex_);
  std::deque<IoResult> out;
  out.swap(results_);
  outstanding_ -= out.size();
  return out;
}
void IoPool::work() {
  for (;;) {
    IoJob job;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !jobs_.empty(); });
      if (stop_ || jobs_.empty())
        return;
      job = std::move(jobs_.front());
      jobs_.pop_front();
    }
    IoResult result;
    result.id = job.id;
    result.epoch = job.epoch;
    result.kind = job.kind;
    try {
      if (job.kind == IoJob::Kind::open)
        result.resource = open_resource(job.file.get(), job.target);
      else {
        std::array<char, block_bytes> bytes{};
        ssize_t n;
        do {
          n = pread(job.file.get(), bytes.data(), std::min(job.length, bytes.size()),
                    static_cast<off_t>(job.offset));
        } while (n < 0 && errno == EINTR);
        if (n <= 0)
          result.error = true;
        else
          result.bytes.assign(bytes.data(), static_cast<std::size_t>(n));
      }
    } catch (...) {
      result.error = true;
    }
    {
      std::lock_guard lock(mutex_);
      results_.push_back(std::move(result));
    }
    std::uint64_t one = 1;
    ssize_t sent;
    do {sent=::write(event_.get(),&one,sizeof one);} while(sent<0&&errno==EINTR);
  }
}
void EventLog::start() {
  sink_.reset(::open("/proc/self/fd/2", O_WRONLY | O_APPEND | O_NONBLOCK | O_CLOEXEC));
  worker_ = std::thread([this] { work(); });
}
EventLog::~EventLog() {
  {
    std::lock_guard lock(mutex_);
    stop_ = true;
    queue_.clear();
  }
  cv_.notify_one();
  if (worker_.joinable())
    worker_.join();
}
void EventLog::emit(std::string record) {
  std::lock_guard lock(mutex_);
  if (record.size() > 256 || queue_.size() >= 128 || stop_ || !sink_) {
    ++dropped_;
    return;
  }
  queue_.push_back(std::move(record));
  cv_.notify_one();
}
std::size_t EventLog::dropped() {
  std::lock_guard lock(mutex_);
  return dropped_;
}
void EventLog::finish(std::string record) {
  std::lock_guard lock(mutex_);
  if(record.size()<=256) final_=std::move(record);
}
void EventLog::work() {
  sigset_t mask;
  sigemptyset(&mask);sigaddset(&mask,SIGPIPE);
  pthread_sigmask(SIG_BLOCK,&mask,nullptr);
  for (;;) {
    std::string record;
    bool final=false;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
      if(stop_) {record=std::move(final_);final=true;}
      else {record=std::move(queue_.front());queue_.pop_front();}
    }
    // Pipe/socket sinks are nonblocking. Regular-file writes run only here and
    // depend on the operator's storage; a stuck kernel filesystem cannot safely
    // be cancelled by detaching a C++ thread.
    if(record.empty()) return;
    pollfd fd{sink_.get(), POLLOUT, 0};
    if (poll(&fd, 1, 50) > 0) {
      auto ignored = ::write(sink_.get(), record.data(), record.size());
      (void)ignored;
    }
    if(final) return;
  }
}
} // namespace uf
