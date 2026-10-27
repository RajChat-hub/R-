# 0.2.0-dev — 27 October 2026

Phase A: replace repeated HTTP buffer parsing with persistent bounded decoding; separate connection state; offload filesystem operations through a fixed two-worker pool and eventfd completions; add bounded asynchronous logging and explicit memory admission.

Phase B/C: original nonblocking native HTTP reverse proxy; configured numeric IPv4 upstreams; bidirectional streaming and backpressure; strict response parsing and canonical message framing; controlled Host and forwarding metadata; hop-by-hop stripping; gateway errors and independent deadlines; no pooling/retries.

Phase D: preserve static regressions and add independent proxy integration, response-parser unit/fuzz coverage, streaming and memory/resource cases, demo application/configuration and paired development benchmark. See PHASE2.md for observed validation status and limitations. Remains a development milestone.

# 0.1.0-dev — 27 October 2026

Original C++20 Linux HTTP/static-routing foundation. Historical validation is retained in VALIDATION.md. No unrelated repository entry was changed, and foundation PR #1 remains unmerged.
