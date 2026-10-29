# UNFALTERING 0.3.0-dev feature status

**Development milestone only; public production release is not approved.**

| Area | Actual state |
|---|---|
| Original Linux 64-bit epoll / HTTP server | Implemented; original framing, static, virtual-host, routing and CLI regressions retained |
| Native HTTP/HTTPS reverse proxy | Streaming, configured numeric IPv4, one fresh connection per transaction, no pooling or retries |
| TLS listener | OpenSSL TLS 1.2/1.3; nonblocking handshake/IO, bounded close, one operator-configured certificate |
| Verified TLS upstream | Chain and SAN identity verification, explicit DNS SNI or numeric-IP SAN policy; system/custom CA stores; no insecure bypass |
| Configuration compatibility | Existing HTTP examples retained; validated optional TLS blocks; key/cert/trust preflight |
| Filesystem workers | Two workers / 64 total credits; pending jobs cancelled on shutdown; active syscalls remain joined |
| Logging | Bounded lossy diagnostics; saturation and blocked-pipe tests; never a durable audit trail |
| Test / sanitizer / fuzz / concurrency evidence | See PHASE3.md and evidence/phase3; bounded development campaigns, not security certification |
| HTTP/2 / HTTP/3 / tunnels / WebSockets | Not implemented |
| DNS / cache / load balancing / health pools | Not implemented |
| PHP-FPM / FastCGI | Not implemented |
| Multi-listener / multi-certificate SNI / mTLS / ACME / live reload | Not implemented |
| IOCP / kqueue / non-Linux / multi-reactor distribution / WAF | Not implemented |
| Commercial distribution / general availability | Not approved; first-party licensing pending |

Phase 3 draft PR #3 targets Phase 2, which remains stacked on foundation PR #1. None is merged automatically. No public deployment or binary release occurred. Existing jdk, archive pointer and script.rsharp remain unchanged.

Operational limits: trusted filesystem calls and regular-file logging can still block process joining indefinitely; use external process supervision. TLS admission reservations are not hard OpenSSL/RSS/cgroup limits. Context/trust-store allocations and kernel socket memory need headroom. Pending TLS writes serialize further TLS IO and can reach deadlines with an uncooperative bidirectional peer. Unsupported Expect, chunk extensions and trailers remain rejected. App effects before disconnect or invalid body cannot be rolled back. See operations/TLS.md and the updated threat model.

Release gates remain: independent security/protocol and dependency review, longer-duration/multi-day operational testing, certificate renewal/key-management procedures, licensing approval, installation/upgrade/rollback packaging, and explicit release authorization. Passing CI does not establish public production readiness or superiority over another server.
