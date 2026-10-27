# Phase 2 development threat model

**Not approved for public production deployment.** No independent security audit or exhaustive HTTP conformance review has occurred.

## Trust boundaries

Clients and upstream HTTP bytes are untrusted. Listener/upstream configuration and document-root contents are operator-controlled. Numeric upstream addresses and Host values are validated before listening; route references are fixed configuration, never derived from client URLs, authority or forwarding headers. There is no DNS lookup on the reactor, arbitrary-destination forward-proxy mode, TLS, WAF, authentication or route-level authorization. Unknown valid hosts use the configured default static root.

## HTTP and forwarding

Persistent bounded state machines reject duplicate Content-Length (including equal duplicates), CL+TE, unsupported transfer codings, invalid controls, obs-fold and malformed chunk delimiters. Headers/counts, decoded body length and all wire queues are bounded. Expect: 100-continue, extensions and nonempty trailers are unsupported and rejected. Upstream 101 and CONNECT tunnels are refused. HEAD/204/304 bodies are not forwarded.

Connection-nominated and standard hop-by-hop headers are stripped in both directions. Framing is reconstructed rather than copied blindly. Upstream Host is configured; incoming Forwarded, X-Forwarded-* and X-Real-IP are removed. The optional replacement policy uses only the directly accepted peer, validated request authority and the actual plaintext listener protocol. Do not enable application trust in arbitrary custom header aliases.

Each transaction uses a fresh upstream connection with Connection: close, no pooling and no automatic retries. Requests may reach an application before a later malformed chunk or client disconnect is detected; aborting transport does not roll back application side effects. Application-level idempotency is still required.

Response boundary errors observed before commitment yield 502; connect/response/send timeouts yield 504. Once committed, an error closes the response without a forged success terminator or second HTTP response. Buffered bytes beyond a fixed-length boundary are rejected; any unread trailing bytes are discarded with that non-reused upstream socket. Informational responses are consumed with a count cap.

## Filesystem, resources and shutdown

Descriptor-relative openat/O_NOFOLLOW traversal and regular-file checks preserve the original encoded traversal and symlink-escape defenses. Jobs own duplicated descriptors and carry connection/epoch identities. IO workers are fixed at two; their queued/active/completed jobs share 64 credits. Event notifications are nonblocking. Late results cannot address a reused descriptor or newer request.

Each wire queue is capped at buffer_bytes. Connection admission uses max_connections and a conservative memory reservation (documented in PROXY.md); allocator/runtime overhead, fixed queues, thread stacks and kernel socket buffers require separate OS budgets. A 128 MiB stream is tested with bounded wire queues and a noninstrumented child-RSS threshold. Instrumented RSS is reported separately because sanitizer quarantine has different memory costs.

The reactor never performs static open/stat/pread or sink writes. Logger capacity is fixed and records may be dropped. Pipe delivery is nonblocking and SIGPIPE is blocked in its worker. Log records exclude request bodies, query strings and credentials. A final diagnostic is best-effort, not a durable audit guarantee.

Connection and upstream identities, explicit readiness masks, write/read budgets, deadlines and deregistration during full-buffer backpressure prevent the tested descriptor-reuse, half-close, resource-exhaustion and shutdown failures. A reproduced full-queue wake-up deadlock was corrected by pumping retained upstream data after a client write budget frees output space.

Shutdown stops accepts, bounds network draining, closes both endpoints and joins workers. Kernel operations permanently stuck on hostile or failed regular-file storage are not safely cancellable; joining may exceed the network deadline. Use trustworthy local storage and an external service-manager termination policy.

## Release limits

No unimplemented TLS, HTTP/2/3, cache, load balancing, FastCGI, multi-reactor, live reload, WAF or non-Linux security capability is claimed. ASan/UBSan and bounded fuzz smoke checks are evidence for tested cases, not proof against all input/resource attacks. ThreadSanitizer, long-duration soak testing, expanded fuzzing, independent review and production operating procedures remain future gates.
