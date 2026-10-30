// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
#include <openssl/ssl.h>
#include <array>
#include <memory>
#include <sys/types.h>
namespace uf {
struct ContextDeleter {void operator()(SSL_CTX* p) const {SSL_CTX_free(p);}};
using TlsContext=std::unique_ptr<SSL_CTX,ContextDeleter>;
TlsContext listener_context(const Config& cfg);
TlsContext listener_context(const ListenerTls& tls);
TlsContext upstream_context(const Upstream& upstream);
void validate_tls(const Config& cfg);
class TlsRegistry {
  ListenerTls policy_;
  TlsContext default_;
  std::map<std::string,TlsContext> sites_;
  static int select(SSL* ssl,int* alert,void* argument) noexcept;
public:
  explicit TlsRegistry(const ListenerTls& policy);
  SSL_CTX* context() const {return default_.get();}
  bool allows_host(SSL* ssl,std::string_view host) const;
};
class Tls {
  SSL* ssl_=nullptr;
  const TlsRegistry* registry_=nullptr;
  std::array<char,16384> write_bytes_{};
  std::array<char,8192> read_bytes_{};
  std::size_t write_size_=0,read_size_=0,read_offset_=0;
  std::uint32_t read_wait_=0,write_wait_=0,control_wait_=0;
  bool read_cross_=false,ready_=false;
  int result(int rc,std::uint32_t& wait);
public:
  unsigned long error=0;
  long verification=0;
  Tls(SSL_CTX* context,int fd,const Upstream* upstream=nullptr,const TlsRegistry* registry=nullptr);
  ~Tls();
  Tls(const Tls&)=delete;
  Tls& operator=(const Tls&)=delete;
  bool allows_host(std::string_view host) const {return !registry_||registry_->allows_host(ssl_,host);}
  bool ready() const {return ready_;}
  int handshake(); // 1 ready, 0 WANT, -1 fatal.
  int shutdown(); // 1 closed, 0 WANT, -1 fatal.
  ssize_t read(char* data,std::size_t size);
  ssize_t write(const char* data,std::size_t size);
  std::uint32_t interest(std::uint32_t application) const;
  bool readable(std::uint32_t events) const;
  bool writable(std::uint32_t events) const;
  bool pending() const;
  bool write_pending() const {return write_size_!=0;}
};
}
