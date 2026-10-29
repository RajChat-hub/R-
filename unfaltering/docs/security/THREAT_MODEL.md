# Phase 3 development threat model

**Not approved for public production deployment.** No independent security audit or exhaustive HTTP conformance review has occurred.

## Trust boundaries

Clients and upstream HTTP bytes are untrusted. Listener/upstream configuration and document-root contents are operator-controlled. Numeric upstream addresses and Host values are validated before listening; route references are fixed configuration, never derived from client URLs, authority or forwarding headers. There is no DNS lookup on the reactor, arbitrary-destination forward-proxy mode, WAF, authentication or route-level authorization. Unknown valid hosts use the configured default static root.

## HTTP and forwarding

Persistent bounded state machines reject duplicate Content-Length (including equal duplicates), CL+TE, unsupported transfer codings, invalid controls, obs-fold and malformed chunk delimiters. Headers/counts, decoded body length and all wire queues are bounded. Expect: 100-continue, extensions and nonempty trailers are unsupported and rejected. Upstream 101 and CONNECT tunnels are refused. HEAD/204/304 bodies are not forwarded.

Connection-nominated and standard hop-by-hop headers are stripped in both directions. Framing is reconstructed rather than copied blindly. Upstream Host is configured; incoming Forwarded, X-Forwarded-* and X-Real-IP are removed. The optional replacement policy uses only the directly accepted peer, validated request authority and the actual native listener protocol (http or https). Do not enable application trust in arbitrary custom header aliases.

Each transaction uses a fresh upstream connection with Connection: close, no pooling and no automatic retries. Requests may reach an application before a later malformed chunk or client disconnect is detected; aborting transport does not roll back application side effects. Application-level idempotency is still required.

Response boundary errors observed before commitment yield 502; connect/response/send timeouts yield 504. Once committed, an error closes the response without a forged success terminator or second HTTP response. Buffered bytes beyond a fixed-length boundary are rejected; any unread trailing bytes are discarded with that non-reused upstream socket. Informational responses are consumed with a count cap.

## Filesystem, resources and shutdown

Descriptor-relative openat/O_NOFOLLOW traversal and regular-file checks preserve the original encoded traversal and symlink-escape defenses. Jobs own duplicated descriptors and carry connection/epoch identities. IO workers are fixed at two; their queued/active/completed jobs share 64 credits. Event notifications are nonblocking. Late results cannot address a reused descriptor or newer request.

Each wire queue is capped at buffer_bytes. Connection admission uses max_connections and a conservative memory reservation (documented in PROXY.md); allocator/runtime overhead, fixed queues, thread stacks and kernel socket buffers require separate OS budgets. A 128 MiB stream is tested with bounded wire queues and a noninstrumented child-RSS threshold. Instrumented RSS is reported separately because sanitizer quarantine has different memory costs.

The reactor never performs static open/stat/pread or sink writes. Logger capacity is fixed and records may be dropped. Pipe delivery is nonblocking and SIGPIPE is blocked in its worker. Log records exclude request bodies, query strings and credentials. A final diagnostic is best-effort, not a durable audit guarantee.

Connection and upstream identities, explicit readiness masks, write/read budgets, deadlines and deregistration during full-buffer backpressure prevent the tested descriptor-reuse, half-close, resource-exhaustion and shutdown failures. A reproduced full-queue wake-up deadlock was corrected by pumping retained upstream data after a client write budget frees output space.

Shutdown stops accepts, bounds network draining, closes both endpoints and joins workers. Kernel operations permanently stuck on hostile or failed regular-file storage are not safely cancellable; joining may exceed the network deadline. Use trustworthy local storage and an external service-manager termination policy.

## Release limits

No unimplemented HTTP/2/3, cache, load balancing, FastCGI, multi-reactor, live reload, WAF or non-Linux security capability is claimed. ASan/UBSan and bounded fuzz smoke checks are evidence for tested cases, not proof against all input/resource attacks. Focused ThreadSanitizer execution, timed development drills and expanded fuzzing are recorded in PHASE3.md. These do not replace independent review, multi-day operational soak tests or production operating procedures.

Connection-option tokens are capped by header_count across the entire header section, preventing a single comma-heavy Connection field from allocating an unbounded hop-header set. The current build explicitly requires 64-bit Linux; no 32-bit wire-counter compatibility is claimed.

## TLS trust and attack surface

OpenSSL 3 handles TLS; original epoll still owns scheduling. Only TLS 1.2/1.3 with security level 2 and ECDHE/AEAD defaults are enabled. Renegotiation, compression, tickets/session cache and 0-RTT are disabled. ALPN selects HTTP/1.1 only. The listener has one operator-configured certificate; virtual hosts must be covered by it. Certificate/key/trust-file loading occurs before accept. Malformed, mismatched or invalid-date listener leaf material fails closed. No private key, real credential, custom cipher or insecure verification bypass is committed.

Upstream certificate chains and SAN identity are mandatory; explicitly configured DNS SNI must match verify_name, or numeric-IP SAN verification uses sni off. The connected address remains numeric/operator-controlled, independent of HTTP Host. Invalid/expired peers and handshake failures yield 502; connect/handshake deadline expiry yields 504. Peer certificate chains are limited to 64 KiB and depth 8. OpenSSL's uninstrumented distribution libraries remain a separate security boundary; application sanitizer passes do not audit the library itself. Package maintenance/renewal, CA selection and key access protection are operator responsibilities.

SSL objects are confined to the reactor; WANT_READ/WANT_WRITE retries use bounded stable buffers and explicit readiness, handshake/operation/close deadlines. TLS state adds conservative admission reservations and separate tested RSS ceilings, not a hard OpenSSL allocation guarantee. Context/trust-store memory and OS/socket budgets are separate. Slow/incomplete handshakes consume bounded admitted slots only until their absolute deadline. Graceful downstream close has a bounded close_notify exchange; malformed, reset or shutdown-deadline paths may force close. An upstream that omits TLS close_notify cannot successfully terminate a close-delimited HTTP response. Authenticated bytes queued before a transport error may complete a validated fixed/chunked boundary; they do not authorize extra data, forged terminators or success for incomplete framing.

Structured TLS diagnostics expose only numeric codes/direction, not secrets. Logs remain lossy and are not an audit trail. IO shutdown now cancels queued jobs but cannot safely terminate active kernel calls; external process supervision and trusted local storage remain necessary. No public production readiness, independent cryptographic review, FIPS validation, certificate automation, live reload, mutual TLS or multi-certificate SNI selection is claimed.
