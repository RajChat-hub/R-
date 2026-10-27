# UNFALTERING 0.2.0-dev feature status

| Area | Actual state |
|---|---|
| Linux native epoll / HTTP static server | Implemented; original regression suite retained |
| Stateful bounded HTTP decoding | Implemented for requests and upstream responses; sink backpressure and fragmentation tested |
| Filesystem/log offloading | Two fixed IO workers; 64 total outstanding work credits; bounded lossy logger |
| Memory/deadline limits | Per-queue caps, conservative connection reservations, connect/upload/send/response/client-write/drain deadlines |
| Virtual hosts / routes / CLI | Existing valid example retained; root and configured proxy routes supported |
| Native HTTP upstream | Implemented; configured numeric IPv4, one connection per transaction, no retries/pooling |
| Streaming / framing / headers | Fixed/chunked/EOF response handling; strict framing; controlled Host and forwarding metadata |
| Test and fuzz evidence | See PHASE2.md and evidence directory; native GCC/Clang CI is the final execution route after workspace loss |
| TLS / HTTPS upstreams | Not implemented |
| HTTP/2 / HTTP/3 / tunnels | Not implemented |
| Cache / load balancing / health pools | Not implemented |
| PHP-FPM / FastCGI | Not implemented |
| IOCP / kqueue / non-Linux support | Not implemented |
| Multi-reactor distribution / reload / WAF | Not implemented |
| Commercial distribution / general availability | Not approved |

Known operational limits: regular-file worker calls can block on pathological storage and delay thread joining; logger drops records on saturation and is not an audit trail; memory reservations are not a hard process RSS limit; timeouts bound finite streams rather than providing unlimited SSE sessions. Strict unsupported protocol forms are rejected. No Nginx capacity or superiority claim is made.

Next engineering gates: sustained stress and fuzz campaigns, ThreadSanitizer where executable, independent protocol/security review, operational failure drills, and then separately implemented TLS. Review the stacked Phase 2 draft without merging the foundation automatically. No public deployment or binary release has occurred.
