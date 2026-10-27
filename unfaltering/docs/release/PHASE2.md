# Phase 2 engineering record

Base: 549c531e9ce2eff7fa5d5c64efc5d8fefce89c00. Separate branch: feat/unfaltering-phase2-proxy-20261027. Foundation PR #1 remains unmerged.

## Phase A gate (completed before adding proxy code)

GCC 13 Debug and GCC Debug ASan/UBSan each passed all 8 CTest groups. The original 22 socket tests remain; added adversarial tests cover 4,000 tiny fragmented chunks, a blocked logging pipe, and memory-reservation admission. Unit checks feed a 64 KiB body one byte at a time, verify consumed-wire accounting, enforce a full sink's no-progress boundary and verify the 64-job worker limit. Test output is preserved in evidence/phase2/phase-a-*.log. Local leak detection is disabled because of the previously documented sandbox /proc limitation.

The reactor retains one StreamParser per connection. Header lines, fixed bodies, chunk sizes/data/delimiters and final empty trailers advance through explicit states, with no repeated parsing of a retained incomplete request. Metadata is bounded; decoded payload is drained to a bounded caller sink, not accumulated per request. The one-shot parse_request helper remains for compatibility/tests, not the reactor path.

Connection state is separated into connection.hpp. Exactly two IO workers handle descriptor-relative open/stat and 16 KiB pread jobs. Queued, active and completed work together are limited to 64 credits; completion notifications use eventfd. Jobs own duplicated descriptors and carry connection/epoch identities so disconnects, descriptor reuse and late results cannot target a newer request. Queue overload produces 503 or defers reads with backpressure.

Request logging uses one separate fixed-capacity, lossy queue and worker; the reactor never writes to its sink. Pipe sinks are opened nonblocking and use bounded poll waits. A full queue drops records. Metadata and logs exclude request bodies, credentials and query strings. A permanently stuck regular-file kernel operation is not forcibly cancellable: trustworthy local storage is required, and worker joining can outlast the network drain deadline if the filesystem itself hangs. This is a documented operational limit, not a hard real-time guarantee.

buffer_bytes bounds each connection wire buffer; memory_budget_bytes caps conservative connection reservations in addition to max_connections. These are application admission bounds, not a substitute for an OS RSS/cgroup limit; allocator, thread stacks, fixed queues and kernel socket memory must also be budgeted by the operator.

An asynchronous-file pipelining regression found during the refactor was fixed: waiting for a worker completion no longer becomes a zero-byte send and premature disconnect. All existing keep-alive, half-close, blocked-writer, descriptor-exhaustion and shutdown tests passed after that fix.

## Remaining at this checkpoint

Native reverse proxy, upstream configuration, response-parser fuzzing and the full Phase B/C/D acceptance suite are not yet implemented at this Phase A checkpoint. No production-readiness, TLS, HTTP/2/3, cache, PHP-FPM or load-balancing claim is made.
