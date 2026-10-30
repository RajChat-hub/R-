// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/config.hpp"
#include "unfaltering/site.hpp"
#include <arpa/inet.h>
#include <algorithm>
#include <charconv>
#include <fstream>
#include <set>
#include <stdexcept>
#include <vector>
namespace uf {
namespace {
struct Lex {
  std::string input;
  std::size_t pos = 0, line = 1;
  std::string word;
  [[noreturn]] void error(const std::string &message) const {
    throw std::runtime_error("config line " + std::to_string(line) + ": " + message);
  }
  void next() {
    word.clear();
    while (pos < input.size()) {
      char c = input[pos];
      if (c == '#') {
        while (pos < input.size() && input[pos] != '\n')
          ++pos;
        continue;
      }
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
        if (c == '\n')
          ++line;
        ++pos;
        continue;
      }
      break;
    }
    if (pos == input.size())
      return;
    char c = input[pos++];
    if (c == '{' || c == '}' || c == ';') {
      word = c;
      return;
    }
    if (c == '"') {
      while (pos < input.size() && input[pos] != '"') {
        if (input[pos] == '\n' || input[pos] == '\r' || input[pos] == '\\' ||
            static_cast<unsigned char>(input[pos]) < 32)
          error("invalid quoted value");
        word += input[pos++];
      }
      if (pos == input.size())
        error("unterminated quote");
      ++pos;
      return;
    }
    word += c;
    while (pos < input.size()) {
      c = input[pos];
      if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == ';' || c == '{' || c == '}' ||
          c == '#')
        break;
      word += c;
      ++pos;
    }
  }
  void need(const std::string &s) {
    if (word != s)
      error("expected '" + s + "', found '" + word + "'");
    next();
  }
  std::string value() {
    if (word.empty() || word == "{" || word == "}" || word == ";")
      error("expected value");
    auto v = word;
    next();
    need(";");
    return v;
  }
};
unsigned numeric(Lex &l, unsigned min, unsigned max) {
  auto v = l.value();
  unsigned n = 0;
  auto [p, e] = std::from_chars(v.data(), v.data() + v.size(), n);
  if (e != std::errc{} || p != v.data() + v.size() || n < min || n > max)
    l.error("numeric limit out of range");
  return n;
}
} // namespace
Config read_config(const std::filesystem::path &file) {
  std::ifstream f(file, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot open config");
  std::string text;
  char buffer[4096];
  while (f.read(buffer, sizeof buffer) || f.gcount()) {
    text.append(buffer, static_cast<std::size_t>(f.gcount()));
    if (text.size() > 65536)
      throw std::runtime_error("config exceeds 64 KiB");
  }
  return parse_config(text, file);
}
Config parse_config(std::string_view text, const std::filesystem::path &file) {
  if (text.size() > 65536)
    throw std::runtime_error("config exceeds 64 KiB");
  Lex l{std::string(text), 0, 1, {}};
  Config c;
  std::set<std::string> keys;
  l.next();
  l.need("server");
  l.need("{");
  auto path = [&](const std::string &s) {
    auto p = std::filesystem::path(s);
    if (p.is_relative())
      p = std::filesystem::absolute(file).parent_path() / p;
    if (!std::filesystem::is_directory(p))
      l.error("root is not a directory");
    return p.lexically_normal();
  };
  auto file_path = [&](const std::string& value) {
    auto p=std::filesystem::path(value);
    if(p.is_relative())p=std::filesystem::absolute(file).parent_path()/p;
    return p.lexically_normal();
  };
  auto identity = [&](const std::string& value) {
    if(value.empty()||value.size()>253||!valid_host(value)||value.find(':')!=value.npos||value.find('*')!=value.npos)
      l.error("TLS identity requires literal DNS name or numeric IPv4 without port");
  };
  auto parse_listener_tls = [&](ListenerTls& tls) {
      tls.enabled=true;l.need("{");std::set<std::string> tls_keys;
      while(l.word!="}") {
        auto field=l.word;
        if(field.empty()||(field!="site"&&!tls_keys.insert(field).second)) l.error("duplicate or incomplete listener TLS option");
        l.next();
        if(field=="site") {
          auto name=dns_identity(l.word);
          if(!name||tls.sites.contains(*name)||tls.sites.size()>=32)l.error("invalid, duplicate or excessive TLS site");
          l.next();l.need("{");Certificate cert;std::set<std::string> seen;
          while(l.word!="}") {
            auto option=l.word;if(option.empty()||!seen.insert(option).second)l.error("duplicate/incomplete TLS site");l.next();
            if(option=="certificate")cert.certificate=file_path(l.value());
            else if(option=="private_key")cert.private_key=file_path(l.value());
            else l.error("unknown TLS site field");
          }
          l.need("}");if(cert.certificate.empty()||cert.private_key.empty())l.error("site certificate and key required");
          tls.sites.emplace(*name,std::move(cert));
        } else if(field=="default_name") {
          auto name=dns_identity(l.value());if(!name)l.error("invalid default TLS DNS identity");tls.default_name=*name;
        } else if(field=="unknown_sni"||field=="no_sni"||field=="host_sni") {
          auto value=l.value();
          if(field=="host_sni") {if(value!="strict"&&value!="off")l.error("host_sni requires strict/off");tls.strict_host=value=="strict";}
          else {if(value!="reject"&&value!="default")l.error("SNI policy requires reject/default");if(field=="unknown_sni")tls.unknown_reject=value=="reject";else tls.require_sni=value=="reject";}
        } else if(field=="certificate")tls.certificate=file_path(l.value());
        else if(field=="private_key")tls.private_key=file_path(l.value());
        else if(field=="handshake_ms")tls.handshake_ms=numeric(l,100,60000);
        else if(field=="shutdown_ms")tls.shutdown_ms=numeric(l,100,5000);
        else l.error("unknown listener TLS option");
      }
      l.need("}");
      if(!tls.sites.empty()) {
        if(tls.default_name.empty())l.error("multi-certificate TLS requires explicit default_name");
        if(tls.sites.contains(tls.default_name))l.error("default TLS identity duplicates site");
        if(!tls_keys.contains("unknown_sni"))tls.unknown_reject=true;
        if(!tls_keys.contains("no_sni"))tls.require_sni=true;
        if(!tls_keys.contains("host_sni"))tls.strict_host=true;
      }
      if(tls.strict_host&&tls.default_name.empty())l.error("strict Host/SNI needs default_name");
      if(tls.certificate.empty()||tls.private_key.empty()) l.error("TLS certificate and private_key required");
  };
  auto endpoint = [&](const std::string& value, std::string& bind, unsigned short& port) {
    auto colon=value.find(':');if(colon==value.npos)l.error("numeric IPv4:port required");
    bind=value.substr(0,colon);in_addr ip{};
    if(inet_pton(AF_INET,bind.c_str(),&ip)!=1)l.error("invalid listener IPv4");
    auto v=value.substr(colon+1);unsigned n=0;auto [end,ec]=std::from_chars(v.data(),v.data()+v.size(),n);
    if(ec!=std::errc{}||end!=v.data()+v.size()||n<1||n>65535)l.error("invalid listener port");
    port=static_cast<unsigned short>(n);
  };
  while (l.word != "}") {
    auto key = l.word;
    if (key.empty())
      l.error("missing closing brace");
    l.next();
    if(key=="listener") {
      Listener listener;listener.name=l.word;
      if(listener.name.empty()||listener.name.size()>64||!std::all_of(listener.name.begin(),listener.name.end(),[](char x){return (x>='a'&&x<='z')||(x>='A'&&x<='Z')||(x>='0'&&x<='9')||x=='_'||x=='-';}))l.error("invalid listener name");
      for(const auto& old:c.listeners)if(old.name==listener.name)l.error("duplicate listener name");
      if(c.listeners.size()>=16)l.error("at most 16 listeners");
      l.next();l.need("{");std::set<std::string> seen;
      while(l.word!="}") {
        auto field=l.word;if(field.empty()||!seen.insert(field).second)l.error("duplicate/incomplete listener field");l.next();
        if(field=="address")endpoint(l.value(),listener.bind,listener.port);
        else if(field=="tls")parse_listener_tls(listener.tls);
        else if(field=="max_connections")listener.max_connections=numeric(l,1,4096);
        else l.error("unknown listener field");
      }
      l.need("}");if(!seen.contains("address"))l.error("listener address required");
      c.listeners.push_back(std::move(listener));continue;
    }
    if (key == "host") {
      auto name = lower(l.word);
      if (!valid_host(name) || name.find(':') != name.npos || c.hosts.contains(name))
        l.error("invalid or duplicate virtual host");
      l.next();
      l.need("{");
      l.need("root");
      auto p = path(l.value());
      l.need("}");
      c.hosts.emplace(name, p);
      continue;
    }
    if (key == "upstream") {
      auto name=l.word;
      if(name.empty()||name.size()>64||!std::all_of(name.begin(),name.end(),[](char x){
        return (x>='a'&&x<='z')||(x>='A'&&x<='Z')||(x>='0'&&x<='9')||x=='_'||x=='-';
      })||c.upstreams.contains(name)) l.error("invalid or duplicate upstream");
      l.next(); l.need("{");
      Upstream u;
      std::set<std::string> seen;
      while(l.word!="}") {
        auto field=l.word;
        if(field.empty()||!seen.insert(field).second) l.error("duplicate or incomplete upstream");
        l.next();
        if(field=="tls") {
          u.tls=true;l.need("{");std::set<std::string> tls_keys;
          while(l.word!="}") {
            auto option=l.word;
            if(option.empty()||!tls_keys.insert(option).second) l.error("duplicate or incomplete upstream TLS option");
            l.next();
            if(option=="verify_name") {u.verify_name=l.value();identity(u.verify_name);}
            else if(option=="sni") {auto value=l.value();if(value!="off") {identity(value);u.sni=value;}}
            else if(option=="ca_file") u.ca_file=file_path(l.value());
            else l.error("unknown upstream TLS option");
          }
          l.need("}");
          in_addr numeric_identity{};
          bool ip=inet_pton(AF_INET,u.verify_name.c_str(),&numeric_identity)==1;
          if(u.verify_name.empty()||!tls_keys.contains("sni")||
             (ip&&!u.sni.empty())||(!ip&&u.sni!=u.verify_name))
            l.error("TLS requires verify_name and matching DNS SNI, or sni off for IP identity");
        } else if(field=="address") {
          auto v=l.value();
          auto colon=v.find(':');
          if(colon==v.npos) l.error("upstream address must be numeric IPv4:port");
          u.address=v.substr(0,colon);
          in_addr ip{};
          if(inet_pton(AF_INET,u.address.c_str(),&ip)!=1||ip.s_addr==INADDR_ANY||ntohl(ip.s_addr)>=0xe0000000U)
            l.error("invalid upstream IPv4");
          auto value=v.substr(colon+1);
          unsigned port=0;
          auto [end,ec]=std::from_chars(value.data(),value.data()+value.size(),port);
          if(ec!=std::errc{}||end!=value.data()+value.size()||!port||port>65535) l.error("invalid upstream port");
          u.port=static_cast<unsigned short>(port);
        } else if(field=="host") {
          u.host=l.value();
          if(!valid_host(u.host)) l.error("invalid controlled upstream Host");
        } else if(field=="connect_ms") u.connect_ms=numeric(l,100,60000);
        else if(field=="response_ms") u.response_ms=numeric(l,100,60000);
        else if(field=="write_ms") u.write_ms=numeric(l,100,60000);
        else if(field=="forwarding") {
          auto v=l.value();
          if(v!="replace"&&v!="off") l.error("forwarding requires replace or off");
          u.forwarding=v=="replace";
        } else l.error("unknown upstream field");
      }
      l.need("}");
      if(u.address.empty()) l.error("upstream address required");
      if(u.host.empty()) u.host=u.address+":"+std::to_string(u.port);
      c.upstreams.emplace(name,std::move(u));
      continue;
    }
    if (key == "route") {
      bool exact = l.word == "exact";
      if (!exact && l.word != "prefix")
        l.error("route requires exact or prefix");
      l.next();
      auto target = l.word;
      if (target.empty() || target.front() != '/' || target.find_first_of("%?#\\") != target.npos)
        l.error("route requires a literal absolute path");
      for (const auto &route : c.routes)
        if (route.exact == exact && route.path == target)
          l.error("duplicate route");
      l.next();
      l.need("{");
      Route route{exact,target,{},{}};
      if(l.word=="root") { l.next(); route.root=path(l.value()); }
      else if(l.word=="proxy") { l.next(); route.upstream=l.value(); }
      else l.error("route requires root or proxy");
      l.need("}");
      c.routes.push_back(std::move(route));
      continue;
    }
    if (!keys.insert(key).second)
      l.error("duplicate key '" + key + "'");
    if(key=="tls") { parse_listener_tls(c.tls); } else if (key == "listen") {
      auto address = l.value();
      auto colon = address.find(':');
      if (colon == address.npos)
        l.error("expected numeric IPv4:port");
      c.bind = address.substr(0, colon);
      in_addr ip{};
      if (inet_pton(AF_INET, c.bind.c_str(), &ip) != 1)
        l.error("listen requires numeric IPv4");
      auto s = address.substr(colon + 1);
      unsigned n = 0;
      auto [p, e] = std::from_chars(s.data(), s.data() + s.size(), n);
      if (e != std::errc{} || p != s.data() + s.size() || n < 1 || n > 65535)
        l.error("invalid listen port");
      c.port = static_cast<unsigned short>(n);
    } else if (key == "root")
      c.root = path(l.value());
    else if (key == "max_connections")
      c.max_connections = numeric(l, 1, 4096);
    else if (key == "max_requests")
      c.max_requests = numeric(l, 1, 10000);
    else if (key == "header_bytes")
      c.limits.header_bytes = numeric(l, 256, 65536);
    else if (key == "header_count")
      c.limits.header_count = numeric(l, 1, 200);
    else if (key == "buffer_bytes")
      c.buffer_bytes = numeric(l, 16384, 262144);
    else if (key == "memory_budget_bytes")
      c.memory_budget_bytes = numeric(l, 1048576, 1073741824);
    else if (key == "body_bytes")
      c.limits.body_bytes = numeric(l, 0, 67108864);
    else if (key == "response_body_bytes")
      c.response_body_bytes=numeric(l,0,1073741824);
    else if (key == "idle_ms")
      c.idle_ms = numeric(l, 100, 60000);
    else if (key == "request_ms")
      c.request_ms = numeric(l, 100, 60000);
    else if (key == "write_ms")
      c.write_ms = numeric(l, 100, 60000);
    else if (key == "shutdown_ms")
      c.shutdown_ms = numeric(l, 100, 30000);
    else
      l.error("unknown key '" + key + "'");
  }
  l.need("}");
  if (!l.word.empty())
    l.error("trailing content");
  if ((!keys.contains("listen")&&c.listeners.empty()) || c.root.empty())
    l.error("listen and root are required");
  if(!c.listeners.empty()&&(keys.contains("listen")||keys.contains("tls")))l.error("legacy listen/tls cannot mix with listener blocks");
  for(std::size_t i=0;i<c.listeners.size();++i)for(std::size_t j=0;j<i;++j) {
    const auto& a=c.listeners[i];const auto& b=c.listeners[j];
    if(a.port==b.port&&(a.bind==b.bind||a.bind=="0.0.0.0"||b.bind=="0.0.0.0"))l.error("conflicting listener bindings");
  }
  for(const auto& route:c.routes) {
    if(!route.upstream.empty()&&!c.upstreams.contains(route.upstream)) l.error("proxy route references unknown upstream");
  }
  std::size_t identities=0;
  for(const auto& listener:configured_listeners(c))if(listener.tls.enabled)identities+=1+listener.tls.sites.size();
  if(identities>64)l.error("at most 64 listener TLS identities per process");
  if(c.buffer_bytes<2*c.limits.header_bytes+1024) l.error("buffer_bytes must fit twice header_bytes plus 1024 bytes");
  if(8*c.limits.header_bytes+8*c.buffer_bytes+512*c.limits.header_count+32768+
     (uses_tls(c)?524288U:0U)>c.memory_budget_bytes)
    l.error("memory budget cannot admit one connection");
  return c;
}
} // namespace uf
