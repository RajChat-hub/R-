// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include <unistd.h>
#include <utility>
namespace uf {
class Fd {
  int value_ = -1;

public:
  explicit Fd(int v = -1) noexcept : value_(v) {}
  ~Fd() {
    if (value_ >= 0)
      ::close(value_);
  }
  Fd(const Fd &) = delete;
  Fd &operator=(const Fd &) = delete;
  Fd(Fd &&x) noexcept : value_(std::exchange(x.value_, -1)) {}
  Fd &operator=(Fd &&x) noexcept {
    if (this != &x) {
      reset();
      value_ = std::exchange(x.value_, -1);
    }
    return *this;
  }
  void reset(int v = -1) noexcept {
    if (value_ >= 0)
      ::close(value_);
    value_ = v;
  }
  int get() const noexcept {
    return value_;
  }
  explicit operator bool() const noexcept {
    return value_ >= 0;
  }
};
} // namespace uf
