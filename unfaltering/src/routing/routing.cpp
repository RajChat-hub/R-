// Copyright (c) 2026 SalesAutoMate. First-party licensing pending.
#include "unfaltering/routing.hpp"
namespace uf {
const Route *match_route(const std::vector<Route> &routes, std::string_view target) {
  auto path = target.substr(0, target.find('?'));
  const Route *best = nullptr;
  // Routing only matches literal request paths. Encoded alternatives still pass
  // through filesystem decoding and safety checks, but do not match these routes.
  for (const auto &r : routes) {
    if (r.exact && path == r.path)
      return &r;
    if (!r.exact && path.starts_with(r.path) &&
        (r.path.back() == '/' || path.size() == r.path.size() || path[r.path.size()] == '/') &&
        (!best || r.path.size() > best->path.size()))
      best = &r;
  }
  return best;
}
} // namespace uf
