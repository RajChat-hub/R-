# 0.2.0-dev — 27 October 2026

Phase A: replace repeated HTTP buffer parsing with persistent bounded decoding; separate connection state; offload filesystem operations through a fixed two-worker pool and eventfd completions; add bounded asynchronous logging and explicit memory admission.

Phase B/C: original nonblocking native HTTP reverse proxy; configured numeric IPv4 upstreams; bidirectional streaming and backpressure; strict response parsing and canonical message framing; controlled Host and forwarding metadata; hop-by-hop stripping; gateway errors and independent deadlines; no pooling/retries.

Phase D: preserve static regressions and add independent proxy integration, response-parser unit/fuzz coverage, streaming and memory/resource cases, demo application/configuration and paired development benchmark. See PHASE2.md for observed validation status and limitations. Remains a development milestone.

# 0.1.0-dev — 27 October 2026

Original C++20 Linux HTTP/static-routing foundation. Historical validation is retained in VALIDATION.md. No unrelated repository entry was changed, and foundation PR #1 remains unmerged.

## 0.3.0-dev — Phase 3

Native epoll-scheduled OpenSSL listener/verified upstream TLS 1.2/1.3, explicit SNI/SAN/CA configuration, bounded handshake/close and stable partial-IO retries; authenticated queued-tail TLS EOF regression fixed. Queued filesystem jobs cancel at shutdown. Added focused concurrency/TSan checks, TLS acceptance/demo/partial-IO/soak/resource tests and differential fragmented-parser fuzzing. Licensing and public production release remain pending. See release/PHASE3.md and operations/TLS.md.
