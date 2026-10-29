# UNFALTERING™ Web Server

**A SalesAutoMate™ Technology — Built to Endure. Engineered to Serve.**

Version 0.3.0-dev. Original C++20 Linux HTTP server and native HTTP/HTTPS reverse-proxy foundation. **Development software, not approved for public production deployment.** First-party licensing remains pending owner approval; no licence grant follows from public repository visibility.

## Build and test (Linux)

Requires CMake 3.20+, GCC 13+ or compatible Clang, POSIX threads, security-maintained OpenSSL 3 development libraries and Python 3 for independent tests. No prebuilt networking or server framework is used.

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Release
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure
./unfaltering/build/unfaltering --check --config unfaltering/examples/server.ufc
./unfaltering/build/unfaltering serve --config unfaltering/examples/server.ufc
```

Open http://127.0.0.1:8080/ after starting the process. For a real local upstream application:

```sh
# Terminal 1
python3 unfaltering/examples/upstream.py
# Terminal 2
./unfaltering/build/unfaltering --check --config unfaltering/examples/proxy.ufc
./unfaltering/build/unfaltering serve --config unfaltering/examples/proxy.ufc
# Terminal 3
curl -i http://127.0.0.1:8080/app/hello
curl -i -X POST --data 'hello upstream' http://127.0.0.1:8080/app/echo
```

## Implemented and bounded

Original stateful HTTP/1.x request/response parsing, Linux epoll lifecycle, static GET/HEAD, virtual hosts, exact/prefix routes, descriptor-relative path protection, and native configured HTTP proxying. Request and response bodies stream through bounded queues, with canonical framing, backpressure, strict hop-by-hop stripping, controlled Host/forwarding policy, timeouts, gateway errors and no retries or upstream pooling. Downstream keep-alive and sequential pipelines remain supported.

Static filesystem work runs in two bounded workers; request logging uses a separate bounded worker and nonblocking pipe delivery. Connections have explicit memory reservations. Graceful network draining closes both endpoints at its deadline. Permanently stalled kernel filesystem operations can delay worker joining; see the threat model.

## Deliberate limits

Linux only; numeric IPv4 upstreams only. Native TLS listeners and verified TLS upstreams are implemented; see [HTTPS operations and demonstration](docs/operations/TLS.md). No HTTP/2/3, FastCGI/PHP-FPM, cache, load balancing, DNS resolver, WebSocket/CONNECT tunnels, transactional reload, multi-reactor worker distribution, WAF or general availability. Expect: 100-continue, chunk extensions and nonempty trailers are rejected. Informational responses are consumed with a bounded count. Routes are not access controls.

[Proxy configuration and demo](docs/operations/PROXY.md), [architecture](docs/architecture/OVERVIEW.md), [threat model](docs/security/THREAT_MODEL.md), [Phase 3 evidence](docs/release/PHASE3.md), [Phase 2 historical evidence](docs/release/PHASE2.md), [feature status](docs/release/STATUS.md).

The Phase 1 validation record is retained as historical evidence. Phase 2 uses a separate branch and stacked draft PR; PR #1 is not merged automatically.

Phase 3 is a separate stacked draft PR #3 targeting Phase 2. None of PRs #1/#2/#3 are merged automatically. Read the exact TLS scope, dependency policy, memory budgets and remaining release gates in the linked operations and release records.
