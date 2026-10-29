// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/tls.hpp"
#include <openssl/err.h>
#include <openssl/x509v3.h>
#include <arpa/inet.h>
#include <sys/epoll.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <stdexcept>
namespace uf {
namespace {
[[noreturn]] void fail() {ERR_clear_error();throw std::runtime_error("TLS configuration validation failed");}
int no_password(char*,int,int,void*) {return 0;}
TlsContext context(bool server) {
  TlsContext ctx(SSL_CTX_new(server?TLS_server_method():TLS_client_method()));
  if(!ctx||!SSL_CTX_set_min_proto_version(ctx.get(),TLS1_2_VERSION)||
     !SSL_CTX_set_max_proto_version(ctx.get(),TLS1_3_VERSION)) fail();
  SSL_CTX_set_security_level(ctx.get(),2);
  SSL_CTX_set_options(ctx.get(),SSL_OP_NO_COMPRESSION|SSL_OP_NO_RENEGOTIATION|SSL_OP_NO_TICKET);
  SSL_CTX_set_mode(ctx.get(),SSL_MODE_ENABLE_PARTIAL_WRITE|SSL_MODE_RELEASE_BUFFERS);
  SSL_CTX_set_max_cert_list(ctx.get(),65536);
  SSL_CTX_set_read_ahead(ctx.get(),0);
  SSL_CTX_set_session_cache_mode(ctx.get(),SSL_SESS_CACHE_OFF);
  SSL_CTX_set_num_tickets(ctx.get(),0);
  SSL_CTX_set_max_early_data(ctx.get(),0);
  if(!SSL_CTX_set_cipher_list(ctx.get(),"ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305")||
     !SSL_CTX_set_ciphersuites(ctx.get(),"TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:TLS_AES_128_GCM_SHA256")) fail();
  return ctx;
}
void bounded_file(const std::filesystem::path& p) {
  std::error_code ec;auto size=std::filesystem::file_size(p,ec);
  if(ec||size==0||size>1048576) fail();
}
int alpn(SSL*,const unsigned char** out,unsigned char* len,const unsigned char* in,unsigned int n,void*) {
  const unsigned char proto[]={8,'h','t','t','p','/','1','.','1'};
  unsigned char* selected=nullptr;
  if(SSL_select_next_proto(&selected,len,proto,sizeof proto,in,n)!=OPENSSL_NPN_NEGOTIATED)
    return SSL_TLSEXT_ERR_ALERT_FATAL;
  // SSL_select_next_proto can point into proto; use static storage instead.
  static const unsigned char http[]="http/1.1";*out=http;*len=8;
  return SSL_TLSEXT_ERR_OK;
}
}
TlsContext listener_context(const Config& cfg) {
  if(!cfg.tls.enabled) return {};
  bounded_file(cfg.tls.certificate);bounded_file(cfg.tls.private_key);
  auto ctx=context(true);
  SSL_CTX_set_default_passwd_cb(ctx.get(),no_password);
  if(SSL_CTX_use_certificate_chain_file(ctx.get(),cfg.tls.certificate.c_str())!=1||
     SSL_CTX_use_PrivateKey_file(ctx.get(),cfg.tls.private_key.c_str(),SSL_FILETYPE_PEM)!=1||
     SSL_CTX_check_private_key(ctx.get())!=1) fail();
  X509* cert=SSL_CTX_get0_certificate(ctx.get());
  if(!cert||X509_cmp_current_time(X509_get0_notBefore(cert))>=0||X509_cmp_current_time(X509_get0_notAfter(cert))<=0) fail();
  SSL_CTX_set_alpn_select_cb(ctx.get(),alpn,nullptr);
  return ctx;
}
TlsContext upstream_context(const Upstream& upstream) {
  if(!upstream.tls) return {};
  auto ctx=context(false);
  SSL_CTX_set_verify(ctx.get(),SSL_VERIFY_PEER,nullptr);
  SSL_CTX_set_verify_depth(ctx.get(),8);
  if(upstream.ca_file.empty()) {if(SSL_CTX_set_default_verify_paths(ctx.get())!=1)fail();}
  else {bounded_file(upstream.ca_file);if(SSL_CTX_load_verify_locations(ctx.get(),upstream.ca_file.c_str(),nullptr)!=1)fail();}
  return ctx;
}
void validate_tls(const Config& cfg) {
  auto listener=listener_context(cfg);
  for(const auto& [name,u]:cfg.upstreams) {(void)name;auto ctx=upstream_context(u);}
}
Tls::Tls(SSL_CTX* ctx,int fd,const Upstream* upstream):ssl_(SSL_new(ctx)) {
  if(!ssl_) fail();
  if(SSL_set_fd(ssl_,fd)!=1) {SSL_free(ssl_);ssl_=nullptr;fail();}
  if(upstream) {
    SSL_set_connect_state(ssl_);
    auto param=SSL_get0_param(ssl_);
    X509_VERIFY_PARAM_set_hostflags(param,X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS|X509_CHECK_FLAG_NEVER_CHECK_SUBJECT);
    in_addr ip{};
    int ok=inet_pton(AF_INET,upstream->verify_name.c_str(),&ip)==1?
      X509_VERIFY_PARAM_set1_ip_asc(param,upstream->verify_name.c_str()):
      X509_VERIFY_PARAM_set1_host(param,upstream->verify_name.c_str(),0);
    if(!ok||(!upstream->sni.empty()&&SSL_set_tlsext_host_name(ssl_,upstream->sni.c_str())!=1)) {
      SSL_free(ssl_);ssl_=nullptr;fail();
    }
    const unsigned char protocol[]={8,'h','t','t','p','/','1','.','1'};
    if(SSL_set_alpn_protos(ssl_,protocol,sizeof protocol)!=0) {SSL_free(ssl_);ssl_=nullptr;fail();}
  } else SSL_set_accept_state(ssl_);
}
Tls::~Tls() {SSL_free(ssl_);}
int Tls::result(int rc,std::uint32_t& wait) {
  if(rc>0) {wait=0;return 1;}
  int e=SSL_get_error(ssl_,rc);
  if(e==SSL_ERROR_WANT_READ||e==SSL_ERROR_WANT_WRITE) {
    wait=e==SSL_ERROR_WANT_READ?EPOLLIN:EPOLLOUT;errno=EAGAIN;return 0;
  }
  wait=0;
  if(e==SSL_ERROR_ZERO_RETURN) return 2;
  error=ERR_peek_last_error();verification=SSL_get_verify_result(ssl_);
  ERR_clear_error();errno=EPROTO;return -1;
}
int Tls::handshake() {
  ERR_clear_error();auto rc=SSL_do_handshake(ssl_);auto status=result(rc,control_wait_);
  if(status==1)ready_=true;
  return status==2?-1:status;
}
int Tls::shutdown() {
  ERR_clear_error();int rc=SSL_shutdown(ssl_);
  if(rc==1) {control_wait_=0;return 1;}
  if(rc==0) {control_wait_=EPOLLIN;return 0;}
  auto status=result(rc,control_wait_);return status==2?1:status;
}
ssize_t Tls::read(char* data,std::size_t size) {
  if(!size) {errno=EAGAIN;return -1;}
  if(write_size_) {errno=EAGAIN;return -1;}
  if(read_offset_==read_size_) {
    std::size_t n=0;ERR_clear_error();
    auto rc=SSL_read_ex(ssl_,read_bytes_.data(),read_bytes_.size(),&n);
    auto status=result(rc,read_wait_);read_cross_=read_wait_==EPOLLOUT;
    if(status==2)return 0;
    if(status!=1)return -1;
    read_offset_=0;read_size_=n;
  }
  auto n=std::min(size,read_size_-read_offset_);
  std::memcpy(data,read_bytes_.data()+read_offset_,n);read_offset_+=n;
  return static_cast<ssize_t>(n);
}
ssize_t Tls::write(const char* data,std::size_t size) {
  if(read_cross_) {errno=EAGAIN;return -1;}
  if(!write_size_) {write_size_=std::min(size,write_bytes_.size());std::memcpy(write_bytes_.data(),data,write_size_);}
  std::size_t n=0;ERR_clear_error();
  auto rc=SSL_write_ex(ssl_,write_bytes_.data(),write_size_,&n);
  auto status=result(rc,write_wait_);
  if(status!=1) {if(status==2)errno=EPIPE;return -1;}
  write_size_=0;return static_cast<ssize_t>(n);
}
std::uint32_t Tls::interest(std::uint32_t application) const {
  if(!ready_)return control_wait_?control_wait_:static_cast<std::uint32_t>(EPOLLIN|EPOLLOUT);
  if(write_size_)return write_wait_;
  if(read_cross_)return read_wait_;
  if(control_wait_)return control_wait_;
  return application;
}
bool Tls::readable(std::uint32_t events) const {
  return !write_size_&&(events&(read_cross_?read_wait_:static_cast<std::uint32_t>(EPOLLIN|EPOLLRDHUP|EPOLLHUP)));
}
bool Tls::writable(std::uint32_t events) const {
  return !read_cross_&&(events&(write_size_?write_wait_:static_cast<std::uint32_t>(EPOLLOUT)));
}
bool Tls::pending() const {return !write_size_&&(read_offset_<read_size_||SSL_pending(ssl_)>0);}
}
