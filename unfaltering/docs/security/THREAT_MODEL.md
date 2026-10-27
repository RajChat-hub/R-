# Development threat model and release restrictions

**Not approved for public production deployment.** No independent penetration audit or exhaustive protocol conformance review has occurred.

Untrusted clients cross the TCP, HTTP framing and path-selection boundaries. Configuration and document-root contents are trusted operator inputs. The process should run unprivileged; the default example binds loopback. There is no authentication, authorization, TLS, WAF or sandbox/chroot. Public routes are not access controls. Unknown hosts deliberately use the default site.

Framing policy: HTTP/1.1 only; one valid Host; token header names; no obs-fold or prohibited controls; duplicate Content-Length rejected even if identical; CL+TE rejected; transfer coding must be exactly chunked; bounded decoded body, header count/bytes and total wire buffer. Expect, extensions and nonempty trailers are rejected. Parse errors force connection closure. Responses construct headers from fixed code-controlled strings and file sizes.

Filesystem policy: decode once; reject null/control/backslash, percent left after decoding and dot components; walk with openat/O_NOFOLLOW from a held root; require regular files; no listings. Symlink/root-escape tests cover direct and directory links. Files modified during serving may cause early EOF and connection closure. Mounted filesystems and malicious privileged local administrators are not isolated by this server.

Resource policy: bounded connections, input buffers, body sizes, header sizes/counts and requests per connection; sequential output; partial IO; fixed work budgets; absolute request/write deadlines; idle deadlines; graceful-drain deadline. Descriptor exhaustion consumes one pending accept with a reserve descriptor and pauses listener registration. Max memory depends on configured connections multiplied by bounded buffers, container overhead and kernel socket buffers. The upper connection bound is not a promise of suitable memory use for a given host.

Known hardening gaps: synchronous regular-file IO and stderr logging can stall the single reactor; parser rescanning can cost CPU on deliberately fragmented input; no per-IP rate limit; no multiworker isolation; no HTTP/2/3; no TLS; no transactional reload; no upstream protection because proxying is absent. Connection exception boundaries fail closed, but process-wide allocation exhaustion remains a fatal startup/runtime error. No proven protection against all resource-exhaustion attacks is claimed.

Tests exercise ambiguous framing, fragmented headers/bodies, request reuse, encoded/double-encoded traversal, symlink escapes, slow clients, blocked writers, resets, descriptor exhaustion and shutdown. ASan/UBSan evidence is documented separately; local leak checking is unavailable in the sandbox, while the remote Clang job passed with leak detection enabled. Future release requires fuzzing, independent security review, operational/load validation and supported-platform CI.
