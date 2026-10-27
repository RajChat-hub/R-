# Configuration version 0.2

One `server { ... }` block is required. `listen "numeric-IPv4:port";` and `root "directory";` are required exactly once. More than one listener/server block is not yet supported and fails validation. No reload CLI exists; stop and restart after `--check` succeeds. Never document SIGHUP as a reload operation in this version.

Values terminate with semicolons. Blocks do not. Double-quoted values permit spaces, but no escapes, backslashes or control characters. `#` starts a comment outside strings. Files are limited to 64 KiB. Paths resolve relative to the config file. Roots must exist and open as nonsymlink directories. `--check` opens configured roots but does not reserve the listener port.

| Key | Default | Accepted bounds |
|---|---:|---|
| max_connections | 256 | 1–4096 |
| max_requests | 100 | 1–10000 per connection |
| header_bytes | 16384 | 256–65536 |
| header_count | 100 | 1–200 |
| body_bytes | 1048576 | 0–67108864 decoded streaming bytes |
| idle_ms | 10000 | 100–60000 |
| request_ms | 5000 | 100–60000; absolute from first byte, not extended by trickle data |
| write_ms | 10000 | 100–60000; absolute response deadline |
| shutdown_ms | 3000 | 100–30000 |

Unknown or duplicate scalar keys, invalid numbers and repeated hostname/route identities are errors. IPv4 port is 1–65535. Addresses default to loopback only in the shipped example; a different explicitly configured numeric IPv4 may expose the server.

```text
server {
 listen "127.0.0.1:8080";
 root "www/default";
 host "one.test" { root "www/one"; }
 host "two.test" { root "www/two"; }
 route prefix "/assets" { root "public"; }
 route exact "/assets/special.txt" { root "special"; }
}
```

Unknown valid Host values use the default root. Duplicate/missing/malformed Host headers are refused. Hostnames match case-insensitively and ignore a valid request authority port. IPv6 authorities are unsupported. This is not a strict hostname allowlist.

Routes are global across configured hosts. Exact matches win; otherwise longest prefix wins at a path-segment boundary (`/assets` does not match `/assetsXYZ`). The complete path is retained: `/assets/file.txt` opens `public/assets/file.txt`. Query strings do not participate. Route matching is literal before decoding; percent-encoded alternatives do not select the same route. All selected roots still apply the filesystem security checks. Routing is not an authorization mechanism. GET and HEAD are the only allowed static methods; there is no configurable method extension yet.

Response bodies are never cached (`Cache-Control: no-store`). Proxy declarations and their stricter buffer requirements are documented in [PROXY.md](PROXY.md). TLS, cache and PHP configuration remain unsupported. Run unprivileged with read-only document roots; do not put secrets in any document root.

Phase 2 adds buffer_bytes (default 65536), memory_budget_bytes (default 67108864) and response_body_bytes (default 1073741824). See PROXY.md for bounds, conservative memory reservations and upstream/proxy-route declarations. Global write_ms is an absolute deadline for static responses and a no-progress deadline for proxy client writes; the upstream response_ms bounds the entire finite proxy transaction.
