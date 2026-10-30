# Phase 4 engineering gates — development only

Base: b54dcecff7f3bc60926c6ff1f81ad0a7d2604b90. Branch: feat/unfaltering-phase4-multisite-balancing-20261031. PRs #1–3 were verified draft/unmerged with their expected heads before modifications. The existing engineering directives, reports, architecture, security and operations documentation and implementation/tests were inspected.

Gate A: multiple IPv4 listeners and per-listener TLS/admission, collision rejection, stable event identities, transactional startup and all-listener shutdown implemented. GCC 13 Debug passed all 18 CTest groups in 68.39 seconds, including all Phase 1–3 tests and five new listener methods. See evidence/phase4/gate-a.log.

Gate B (multi-certificate SNI), C (round-robin pools), D (active/passive health) and E (expanded validation) remain incomplete at this checkpoint. No public deployment, binary publication, first-party license grant or PR merge is authorized by this development checkpoint.

Gate B checkpoint: exact SNI registry, certificate-name/key/date preflight, explicit default/unknown/absent-SNI policies and strict Host/SNI enforcement implemented. Six SNI integration methods plus all existing tests passed: 19/19 GCC Debug CTest groups in 70.58 seconds. Wildcard mappings are not implemented. Gate C/D/E remain incomplete at this checkpoint.

Gate C checkpoint: native bounded backend groups, deterministic per-pool round-robin, eligibility filtering and per-backend connection tracking/caps implemented. Six real HTTP/HTTPS distribution/streaming/concurrency/config/no-retry methods plus all regressions passed: 20/20 GCC Debug CTest groups in 71.80 seconds. Least-active, weights and connection reuse are not implemented. Health checks and final expanded validation remain incomplete at this checkpoint.

Gate D checkpoint: bounded native epoll HTTP/verified HTTPS probes; initial unverified exclusion; failure/recovery hysteresis; passive transport/framing observations without application-status quarantine; per-backend probe state; subsequent-request failover without replay; loopback/authority-restricted read-only metrics implemented. Twelve health integration methods plus state-machine unit checks and all earlier regressions passed: 22/22 GCC Debug groups in 78.18 seconds. A failing metrics Host-authority regression was preserved before restricting the endpoint against DNS-rebinding exposure; see metrics-host-before-fix.log. Gate E remains pending.
