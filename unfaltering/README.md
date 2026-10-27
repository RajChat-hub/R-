# UNFALTERING™ Web Server

**A SalesAutoMate™ Technology — Built to Endure. Engineered to Serve.**

Version 0.1.0-dev. Original C++20 HTTP/1.1 implementation with a Linux epoll reactor.
This is a tested development milestone, **not a production-ready Apache/Nginx replacement**.
First-party licensing is pending owner approval; no open-source licence is granted by repository visibility.

## Build and run (Linux)

Requires GCC 13+ or a compatible Clang compiler, CMake 3.20+, and Python 3 for integration tests.
No runtime networking dependencies or third-party server engine are used.

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Debug
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure
./unfaltering/build/unfaltering --version
./unfaltering/build/unfaltering --check --config unfaltering/examples/server.ufc
./unfaltering/build/unfaltering serve --config unfaltering/examples/server.ufc
# In another terminal:
curl -i http://127.0.0.1:8080/
curl -I -H 'Host: example.test' http://127.0.0.1:8080/
```

Send SIGTERM or Ctrl+C to stop. The listener closes immediately; pending writes drain for at most `shutdown_ms`. Idle/incomplete requests close immediately during shutdown.

## Implemented

- Original bounded HTTP/1.1 parser with fragmented input, persistent connections and sequential pipelining.
- Content-Length and strict chunked-body decoding; ambiguous framing rejected and connection closed.
- GET/HEAD static serving, correct lengths, MIME selection, Date and nosniff headers.
- Descriptor-relative filesystem walking with O_NOFOLLOW on every untrusted component; no directory listing.
- Direct nonblocking Linux epoll, partial reads/writes, per-event work budgets, backpressure, request/idle/write deadlines, connection/request limits, descriptor-exhaustion recovery and bounded shutdown.
- Strict `.ufc` configuration, host roots, exact/longest-prefix static routes, unknown/duplicate-key rejection and `--check`.
- Structured event logs omit targets, query strings, headers and bodies.
- Unit, socket, fault-injection and sanitizer tests; HTTP/config libFuzzer harnesses.

## Not implemented

Reverse proxy, TLS, upstream load balancing, cache, PHP-FPM/FastCGI, live reload, worker distribution, rate limiting, metrics endpoint, HTTP/2, HTTP/3, IOCP, kqueue and binary distribution packages. Unsupported config keys fail instead of silently enabling these features.

HTTP/1.0, IPv6 authority/listeners, Expect: 100-continue, chunk extensions and nonempty trailers are rejected. Only origin-form request targets are supported. See [security limits](docs/security/THREAT_MODEL.md) before any external exposure.

See [configuration](docs/operations/CONFIGURATION.md), [architecture](docs/architecture/OVERVIEW.md), [test evidence](docs/release/VALIDATION.md) and [next milestones](docs/release/STATUS.md).
