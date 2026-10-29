// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/io_pool.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
using namespace uf;
int main() {
  auto check=[](bool ok) {if(!ok) throw std::runtime_error("concurrency invariant");};
  {IoPool idle; for(std::size_t i=0;i<IoPool::capacity;++i) {IoJob job;check(idle.submit(std::move(job)));}
   idle.shutdown();IoJob rejected;check(!idle.submit(std::move(rejected)));check(idle.take().empty());}
  IoPool pool;pool.start();
  std::atomic<unsigned> submitted{0},finished{0};
  std::vector<std::thread> producers;
  for(unsigned i=0;i<4;++i) producers.emplace_back([&] {
    for(unsigned j=0;j<1000;) {IoJob job;job.id=j;
      if(pool.submit(std::move(job))) {++submitted;++j;} else std::this_thread::yield();}
    ++finished;
  });
  unsigned completed=0;
  auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  while(finished<4||completed<submitted) {
    completed+=static_cast<unsigned>(pool.take().size());
    check(std::chrono::steady_clock::now()<end);
    std::this_thread::yield();
  }
  for(auto& p:producers) p.join();
  check(submitted==4000&&completed==4000);pool.shutdown();
  EventLog log; // No sink: saturation deterministically records loss without blocking.
  producers.clear();
  for(unsigned i=0;i<4;++i) producers.emplace_back([&] {for(unsigned j=0;j<10000;++j)log.emit("{}\n");});
  for(auto& p:producers)p.join();
  check(log.dropped()==40000);
  std::cout<<"4000 concurrent worker completions; 40000 counted log drops; queued cancellation passed\n";
}
