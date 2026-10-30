// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#pragma once
#include "config.hpp"
#include <optional>
namespace uf {
// Exact ASCII DNS identities only: no wildcard, empty label, terminal dot or IP.
inline std::optional<std::string> dns_identity(std::string_view input) {
  if(input.empty()||input.size()>253)return {};
  std::string name;name.reserve(input.size());std::size_t label=0;bool numeric=true;
  for(char c:input) {
    if(c=='.') {if(!label||name.back()=='-')return {};label=0;name+=c;continue;}
    if(c>='A'&&c<='Z')c=static_cast<char>(c-'A'+'a');
    if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='-')||(!label&&c=='-')||++label>63)return {};
    if(c<'0'||c>'9')numeric=false;
    name+=c;
  }
  if(!label||name.back()=='-'||numeric)return {};
  return name;
}
inline std::optional<std::string> select_site(const ListenerTls& policy,std::string_view sni) {
  if(sni.empty())return policy.require_sni?std::nullopt:std::optional<std::string>(policy.default_name);
  auto name=dns_identity(sni);if(!name)return {};
  if(*name==policy.default_name||policy.sites.contains(*name))return name;
  return policy.unknown_reject?std::nullopt:std::optional<std::string>(policy.default_name);
}
inline bool site_host_matches(const ListenerTls& policy,std::string_view sni,std::string_view host) {
  if(!policy.strict_host)return true;
  auto selected=select_site(policy,sni);auto normalized=dns_identity(host);
  if(!selected||!normalized||*selected!=*normalized)return false;
  return sni.empty()||dns_identity(sni)==selected;
}
}
