# Feature and release status

| Milestone | State | Evidence / remaining work |
|---|---|---|
| 0 Repository and build foundation | Implemented locally | Existing jdk gitlink, archive pointer and script preserved; original source, CMake, CLI, tests and CI definitions added. Remote GCC Debug/Release, Clang sanitizers/leak checks and 10,000-run HTTP/config fuzz smoke checks passed; see VALIDATION.md. |
| 1 HTTP/1.1 static server | Initial Linux subset implemented and tested | epoll, framing, GET/HEAD, persistence, limits, filesystem isolation, partial IO and shutdown. Strict unsupported protocol forms are documented. |
| 2 Configuration and routing | Initial static-routing subset implemented and tested | Strict single-listener config, default/virtual-host roots, global exact and segment-prefix routes. Proxy routes and configurable per-route methods are not implemented. |
| 3 Reverse proxy | Not implemented | Next critical milestone: nonblocking upstream state machine, framing, streaming/backpressure and trust rules. |
| 4 TLS | Not implemented | Select/pin maintained TLS dependency and review licensing before integration. |
| 5 Load balancing / metrics | Not implemented | Needs verified proxy foundation; health and overload tests. |
| 6 Cache / FastCGI | Not implemented | Implement/test policies and actual PHP-FPM execution separately. |
| 7 Commercial distribution | Not implemented | Install target exists; packages/service management and commercial release gates remain. |

Linux x86-64 is the only locally executed platform. macOS and Windows are unsupported and fail configuration explicitly. Runtime is single-threaded; worker distribution and a complete portable IO abstraction remain future work. SIGHUP reload is not supported. There is no publicly deployed service or production URL.

Next: implement a bounded nonblocking HTTP upstream client without retries of non-idempotent requests; add hop-by-hop header filtering, explicit handling of forwarded metadata, upstream timeouts and real fault-injection tests. Preserve all existing HTTP/filesystem tests. Then integrate TLS and operational hardening before public deployment.

General availability requires: full native supported-platform CI, extended fuzzing, independent protocol/security review, overload and long-duration validation, licensing approval, actual packaging/upgrade testing, and resolution of the documented IO/parser/logging limitations. No Apache/Nginx performance comparison has run.
