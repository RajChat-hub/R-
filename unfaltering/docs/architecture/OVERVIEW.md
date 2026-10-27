# Phase 2 architecture

The native runtime remains original C++20 and Linux epoll. Core HTTP, routing, framing and upstream orchestration do not delegate to a server or event framework.

## Ownership and components

- `StreamParser` in src/http/stream.cpp is a persistent bounded request/response state machine. First line, headers, fixed body, chunk size, chunk data, delimiter, empty trailers, EOF body, done and failed are explicit states. It retains metadata only and writes decoded payload into a caller-limited sink. Header completion is a separate boundary so proxying starts before the full request body arrives.
- `Connection` owns the downstream descriptor, parser, bounded wire queues, deadlines, worker epoch and optional `ProxyTransaction`.
- `Reactor` declarations live in reactor.hpp. Client/static lifecycle and epoll dispatch live in src/net/server.cpp. Upstream readiness, streaming and proxy lifecycle live in src/proxy/native.cpp. Header policy/framing serialization lives in src/proxy/headers.cpp.
- `IoPool` has exactly two workers and 64 total queued/active/completed credits. Root traversal, open/stat and 16 KiB pread run off the reactor. Jobs own duplicated descriptors and carry connection plus epoch identities; stale completions are discarded safely.
- `EventLog` owns one worker and at most 128 records of at most 256 bytes. Pipe sinks are opened nonblocking, SIGPIPE is blocked in the logger, and blocked sinks cause record loss rather than reactor stalls. A final shutdown diagnostic is retained separately.

## Proxy lifecycle

Each transaction opens one nonblocking socket to a validated configured numeric IPv4 address. Unique epoll identities map each upstream endpoint to its downstream connection; descriptor numbers are not event identities. The upstream is never pooled or reused. Connection, absolute response and request-send deadlines are separate.

Request headers are validated before any upstream bytes are sent. Fixed and chunked request bodies stream through a bounded decoded scratch buffer into a bounded upstream wire queue. Chunked framing is regenerated, never forwarded unchecked. The request ends at its exact message boundary; already received downstream pipeline bytes stay with that connection.

Response headers and bodies use the same incremental decoder in response mode. Fixed lengths are retained. Chunked and close-delimited upstream bodies are normalized to downstream chunked framing, enabling downstream persistent connections without upstream pooling. Up to eight informational responses are consumed before the final response; 101/upgrade and CONNECT tunnels are unsupported. HEAD, 204 and 304 have no response body. A final response arriving before request upload completes cancels further upload and forces downstream closure after that response.

Both directions pause reads when their destination queues are full. Upstream descriptors with no useful interest are temporarily deregistered so HUP readiness cannot busy-spin. Client write-budget exhaustion explicitly refills output from retained upstream input; this closes the experimentally found both-queues-full wake-up gap.

Gateway failures produce 502 or 504 only before a response is committed. After any response byte is sent, framing errors, resets or timeouts close the stream without a false terminal chunk or appended second HTTP response. Buffered overflow past a fixed-length response is rejected; unread bytes after a completed boundary are discarded when closing that non-reused upstream socket.

## Limits and shutdown

Four wire queues are individually limited to buffer_bytes. Header/count and decoded body limits are independent; decoded bodies are not accumulated. A conservative per-connection reservation is `8*header_bytes + 8*buffer_bytes + 512*header_count + 32768`; admission is bounded by both max_connections and memory_budget_bytes. The factor for queue capacity covers normal string allocation headroom. This is not an OS RSS guarantee: fixed worker/log queues, allocator/runtime overhead, thread stacks and kernel socket buffers also require operator budgets.

Network event work is capped at 64 KiB and file jobs at 16 KiB. The request deadline is absolute; downstream write timeout detects lack of write progress for proxy responses. The configured absolute upstream response deadline also bounds a streamed transaction. SIGTERM/SIGINT stop accepts and drain active work up to shutdown_ms, then close both endpoints. C++ threads are joined rather than detached.

A kernel operation stuck on an untrustworthy filesystem is not safely cancellable by C++ thread teardown. The reactor remains isolated while worker credits are exhausted, but worker joining may exceed the network drain deadline in that pathological case. Logging to regular files has the same trusted-storage requirement. There is no hard real-time shutdown claim.

References: RFC 9110 https://www.rfc-editor.org/rfc/rfc9110.html and RFC 9112 https://www.rfc-editor.org/rfc/rfc9112.html. This is a deliberately strict HTTP/1.x subset, not exhaustive conformance.

An exact write-budget completion is finalized even when no further readiness event will arrive. The completed transaction is reset immediately, allowing its next keep-alive request. The absolute response deadline also covers final downstream queue draining; write-progress alone cannot extend a transaction indefinitely.
