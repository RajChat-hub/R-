# UNFALTERING 0.3.0-dev — Phase 3 engineering and validation record

29 October 2026. **Development milestone, not approved for public production deployment.**

Repository: https://github.com/RajChat-hub/R-
Branch: `feat/unfaltering-phase3-tls-20261029`
Stacked draft: https://github.com/RajChat-hub/R-/pull/3
Base: `6d64fcc4f1d63dc9763ea3b5bb702fa69fd39407` (Phase 2)
Reliability gate: `b7549f68e109873637c0ef831da0f348cbcc4e32`
Validated runtime/harness code: `443ec49d4c903898471584ed537112585c2833cf`
Final validated code and corrected corpus: `433d585be1745327b36c8731772f49c722b0bc56`

AGENTS.md, the complete MASTER_DIRECTIVE.md, PHASE2.md, STATUS.md, THREAT_MODEL.md, architecture and existing tests were read before source changes. A stale unfinished local Phase 2 checkout was preserved; work started from a separate checkout of the verified remote Phase 2 SHA. Unrelated repository entries/history were preserved. PR #3 targets Phase 2, not main; PR #1/#2/#3 remain unmerged. No credentials/private keys are in Git.

## Delivered scope

| Phase | Delivered behavior |
|---|---|
| A Reliability | Pending filesystem-job cancellation at shutdown; fixed worker limits retained; concurrent worker/logger check; counted no-sink drops; blocked logging, original static/proxy regressions and timed disconnect/fragmentation drills |
| B Native TLS | OpenSSL 3 SSL objects on original epoll reactor; TLS 1.2/1.3, ECDHE/AEAD defaults; nonblocking handshake/read/write/close states; stable bounded WANT retries; bounded deadlines; nonsecret diagnostics |
| C Upstream TLS | Mandatory chain/SAN identity validation, explicit matching DNS SNI or IP SAN/sni off, system/custom CA trust, connect deadline includes TLS; existing numeric addresses, framing and no-retry lifecycle retained |
| D Configuration | Optional server/upstream TLS blocks; old HTTP examples retained; startup/--check key/cert/trust validation; strict unknown/duplicate/inconsistent setting rejection |
| E Acceptance | Original HTTP tests plus verified HTTPS static/proxy, GET/POST/chunked/pipelines, TLS protocol versions, identity/expiration/trust errors, slow/incomplete handshakes, abrupt disconnects, TLS EOF/framing, large streaming/backpressure, clean/idle close_notify, resource admission and RLIMIT recovery, active shutdown, exact shipped HTTPS demo, compiler/sanitizer/TSan/fuzz/soak evidence |

No web-server engine, event-loop framework, custom encryption, DNS resolver, pooling, retries, HTTP/2/3, cache, load balancer, FastCGI or public deployment was introduced.

## Phase A gate before TLS implementation

Strict GCC 13 Debug compilation and 22 static / three hardening / 30 proxy methods passed. The first combined shell compilation/test invocation inherited compiler child-RSS accounting and failed only the old child-RSS assertion; a standalone invocation passed without weakening the plaintext ceiling. A 10.0098-second eight-client drill completed 6,454 transactions and 712 injected disconnects. Native worker/logger concurrency test completed 4,000 jobs and counted 40,000 no-sink drops. GCC ThreadSanitizer executed successfully for this component check before TLS integration. Evidence is in `evidence/phase3/reliability*.log` and reliability.txt.

The IO pool now rejects submissions and clears queued, not-started jobs on shutdown. Active kernel calls are joined, never unsafely terminated or detached. This improves queue teardown but cannot guarantee process exit when storage is permanently stuck. The logger remains fixed-capacity/lossy; a blocked pipe test verifies network/shutdown progress, not log durability.

## Actual defect and regression preservation

Initial TLS implementation commit `45d7d8389bba9341aa3bb89138ffd84511fa4a6b` passed local tests but CI Release/Clang runs intermittently truncated the 128-MiB stream. The failing acceptance test was already committed. Local repeated execution reproduced failure after three successful transfers: 134,201,344 decoded bytes with the final 16,384 authenticated bytes still queued behind downstream backpressure. A bare upstream TLS EOF triggered premature gateway cleanup and discarded that retained tail.

`e77ee0f2749375277530eea645c669031977c594` corrected terminal error handling. The parser now drains authenticated queued data first; exact fixed/chunked boundaries can finish, while incomplete or close-delimited bodies still fail without a false success terminator. An errored SSL object is not shut down as though healthy. The acceptance suite adds repeated tail regressions and a separate missing-close_notify EOF-body rejection. Thirty consecutive 128-MiB transfers then passed (3.75 GiB total); native high-water RSS 23,552 KiB, checked queues 131,072 bytes. Failure logs and reproduction output are preserved, not replaced by success-only records.

The TLS fixture's original five-entry accept backlog also produced one short connect-deadline failure under eight concurrent test clients. The independent fixture backlog was raised to 64 and the ordinary test target uses 2,000-ms connect timeout. Explicit 200-ms timeout failure tests remain. This was a fixture contention adjustment, not removal of the failing production behavior test. See tls-initial-fixture-failure.log.

## Final compiler and concurrency execution

Final code push run: https://github.com/RajChat-hub/R-/actions/runs/40924689369
Final code PR run: https://github.com/RajChat-hub/R-/actions/runs/40924697798

Observed compiler jobs passed **17/17 CTest groups**:

- GCC 13.3 Debug: 66.51 seconds.
- GCC 13.3 Release: 65.93 seconds.
- Clang 18.1.3 Debug ASan/UBSan **with leak detection enabled**: 67.64 seconds.

Groups include original 161 assertions, parser/stream units, worker/logger concurrency, 20,000 deterministic mutations, CLI/config, 22 original static methods, three hardening methods, 30 original proxy methods, plaintext demo, native TLS partial IO, shipped HTTPS demo, 20 TLS acceptance methods and timed plaintext/TLS concurrency drills. The native TLS socketpair test forces 5,084 WANT retries through 4-KiB socket send buffers and verifies 4 MiB in both directions plus mutual close_notify.

Remote GCC ThreadSanitizer passed all four selected groups (worker/logger concurrency, native TLS partial IO, TLS soak and plaintext soak): 44.08 seconds CTest / 44.83 seconds including probe. The runtime compatibility probe distinguishes a known unexpected-memory-mapping failure as unsupported/inconclusive, but **this final remote execution actually passed**, rather than taking that fallback. Local TSan also passed these four groups in 43.30 seconds. OpenSSL/system libraries are distribution builds, not fully TSan-instrumented dependencies; these checks are not a whole-system race proof.

Local GCC Debug final suite passed 17/17 in 66.42 seconds. A separate local GCC ASan/UBSan build passed the suite at the preceding queued-tail revision; local leak detection was disabled for the documented sandbox /proc restriction. An initial local sanitizer build produced a zero-byte object and a linker error; deleting only that generated object and rebuilding serially resolved it. The final supported Clang sanitizer/leak evidence is the observed remote CI, not an inferred local result.

## Sanitizer-instrumented fuzz campaigns

The final Clang libFuzzer job passed all three 90-second-budget campaigns; each reported 91 seconds including its final iteration:

| Target | Actual executions | Reported duration |
|---|---:|---:|
| HTTP requests | 711,469 | 91 seconds |
| Configuration including TLS declarations | 783,065 | 91 seconds |
| Upstream responses, including HEAD semantics | 519,140 | 91 seconds |
| Total | 2,013,674 | 273 seconds |

HTTP/request and response harnesses compare variable 1–127-byte fragmentation and 0–128-byte sink capacities against larger slices, checking invariant acceptance, rejection category, bounded counters and successful message boundaries/payload digests. Invalid messages can legitimately expose different valid prefixes before rejection; those prefixes are not required to match. HEAD/no-body, CL/TE conflicts, chunks, trailers and EOF seeds are included. TLS config declarations are fuzzed syntactically without loading arbitrary certificates in the fuzz process. Max input is 65,536 bytes; per-input timeout 3 seconds, process RSS guardrail 768 MiB. Targets compile parser/config sources with Clang fuzzer/address/undefined instrumentation. This does not fuzz the entire OpenSSL engine or replace independent TLS/protocol review.

```sh
cmake -S unfaltering -B unfaltering/build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DUNFALTERING_FUZZ=ON -DBUILD_TESTING=OFF
cmake --build unfaltering/build-fuzz -j2
./unfaltering/build-fuzz/uf_fuzz_http unfaltering/tests/fuzz/seeds/http -max_total_time=90 -max_len=65536 -timeout=3 -rss_limit_mb=768
./unfaltering/build-fuzz/uf_fuzz_config unfaltering/tests/fuzz/seeds/config -max_total_time=90 -max_len=65536 -timeout=3 -rss_limit_mb=768
./unfaltering/build-fuzz/uf_fuzz_response unfaltering/tests/fuzz/seeds/response -max_total_time=90 -max_len=65536 -timeout=3 -rss_limit_mb=768
```

Actual command/output excerpts, job IDs, compiler/dependency revisions, resource diagnostics and PR preservation checks are in evidence/phase3/github-ci.json. Full logs remain in the linked CI jobs. The final code push and PR runs both passed. Later evidence-only commits do not change the validated runtime/test source.

## Extended development drills and resource observations

The final local 120.045-second verified HTTPS drill used eight clients, fragmented requests, slow readers and deliberate abrupt disconnects. It completed **21,455 transactions and 1,946 injected disconnects**; native peak RSS 23,424 KiB, owned network endpoint high-water 17, checked proxy queues 131,072 bytes. This is a two-minute development drill, not a long-running production soak or commercial-capacity benchmark.

CI's 30-second TLS drills also passed: Release 5,776 transactions/520 disconnects; Debug 5,744/520; Clang sanitizer 5,852/528; TSan 4,782/432. Counts depend on runner/client overhead and are not throughput promises.

TLS admission adds 512 KiB per downstream connection for up to two SSL endpoints, in addition to the original reservation. The 32-connection fixture uses a 64-MiB application budget plus explicit test RSS ceilings of 192 MiB native / 768 MiB instrumented. These ceilings cover extra TLS state, contexts, record/certificate buffers and allocator/instrumentation overhead; they do not blindly reuse the plaintext 64-MiB check or enforce an OpenSSL allocator limit. Native Release TLS process peaks were 31,068 KiB (acceptance) and 32,080 KiB (soak); Clang sanitizer acceptance peak was 650,224 KiB, below 786,432 KiB. Owned endpoint high-water was at most 32 in the acceptance suite, bounded by the configured 64-endpoint maximum. Four proxy wire queues remained within 4×65,536 bytes; observed maximum aggregate was 131,072 bytes. The original plaintext limits remain tested independently.

A separate real RLIMIT_NOFILE=32 TLS fixture exhausts descriptors, rejects overload and recovers after clients close. Resource high-water is emitted from inside the native server using getrusage/owned endpoint counts because cross-process /proc paths are unavailable in the local execution environment. Tests do not claim to measure kernel socket memory or all per-thread descriptors.

Local project trees including several incremental compiler builds measured 141,440 KiB at final checks, far below the 16-GiB project target. CI footprints: GCC Release 4,964 KiB, Debug 34,636 KiB, Clang sanitizer 46,112 KiB; benchmark current 2,648 KiB plus foundation 1,320 KiB. No massive toolchain/source downloads or unrelated cleanup occurred.

The retained short plaintext comparison job passed but is not a TLS benchmark, sustained-capacity claim or evidence of superiority over another server. No new commercial performance claim is made.

## Dependency and security policy

Runtime dependency is system OpenSSL 3.x; the API minimum is 3.0, not a frozen security patch level. CI upgraded and recorded Ubuntu OpenSSL `3.0.13-0ubuntu3.16`. Local runtime backport metadata was older and is documented only as test provenance. Operators must use maintained packages/track advisories; the version gate alone cannot certify patch status. Apache-2.0 license material and upstream attribution are included for OpenSSL; first-party licensing remains pending. No vendored TLS source or library binaries are added.

See [TLS operations](../operations/TLS.md), [architecture](../architecture/OVERVIEW.md), [threat model](../security/THREAT_MODEL.md) and [status](STATUS.md) for exact lifecycle, SNI/CA policy, budgets and limitations. Structured TLS logs exclude secret material; logs are lossy and are not an audit trail.

## Reproduce

From repository root on 64-bit Linux:

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Release
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure --verbose
python3 unfaltering/examples/make-test-certs.py unfaltering/examples/tls-local
./unfaltering/build/unfaltering --check --config unfaltering/examples/https.ufc
```

In separate terminals:

```sh
python3 unfaltering/examples/upstream.py --port 9443 --certificate unfaltering/examples/tls-local/server.crt --private-key unfaltering/examples/tls-local/server.key
./unfaltering/build/unfaltering serve --config unfaltering/examples/https.ufc
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 https://localhost:8443/
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 https://localhost:8443/app/hello
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 --data 'hello TLS' https://localhost:8443/app/echo
```

These exact local static/GET/POST flows and forced TLS 1.2/1.3 curl negotiations were executed; output is in evidence/phase3/curl-demo.log. No `-k` bypass is used. The auto-generated test credentials expire and are not public deployment material.

Sanitizers, using separate compiler/build directories:

```sh
cmake -S unfaltering -B unfaltering/build-clang -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DUNFALTERING_SANITIZERS=ON
cmake --build unfaltering/build-clang -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 UNFALTERING_INSTRUMENTED=ON ctest --test-dir unfaltering/build-clang --output-on-failure --verbose
cmake -S unfaltering -B unfaltering/build-tsan -DCMAKE_BUILD_TYPE=Debug -DUNFALTERING_TSAN=ON
cmake --build unfaltering/build-tsan -j2
UNFALTERING_INSTRUMENTED=ON TSAN_OPTIONS=halt_on_error=1 python3 unfaltering/tests/run_tsan.py unfaltering/build-tsan
```

## Remaining release gates

No public production readiness, independent protocol/cryptographic security review, FIPS certification, multi-day soak qualification, exhaustive RFC conformance or finalized binary redistribution/legal approval is claimed. Single listener/certificate per process; no multi-certificate SNI, mTLS, ACME, renewal/reload, DNS, tunnels, HTTP/2/3, load balancing, cache, FastCGI, multi-reactor distribution or non-Linux support.

Pending SSL writes serialize TLS IO and an uncooperative bidirectional peer can reach configured deadlines; arbitrary application tunnelling is not supported. Completed upstream shutdown is best-effort/no pooling. Filesystem syscalls can still delay worker joining indefinitely; external process supervision is required. Memory admission is not a hard RSS/cgroup cap; OS and trust-store overhead remain separate. App effects before late body errors/disconnect cannot be undone. Independent review, operational validation, dependency/key-maintenance procedures, licensing approval and explicit release gates remain mandatory.

## Final corpus byte-preservation check

Final tree comparison found four newly added response seeds had CRLF normalized during transfer. Commit `433d585be1745327b36c8731772f49c722b0bc56` restores their exact wire bytes; compiled runtime and harness source are unchanged. Both push https://github.com/RajChat-hub/R-/actions/runs/40926937688 and PR https://github.com/RajChat-hub/R-/actions/runs/40926950979 passed, including all compiler suites, focused ThreadSanitizer checks and repeated sanitizer fuzz campaigns. The corrected-corpus campaigns completed **875,857 request + 984,976 config + 645,958 response executions = 2,506,791 total**, with 91 seconds each / 273 seconds reported total. The earlier campaign table remains as historical evidence. The final commit after this checkpoint only saves this report and structured evidence; it does not change runtime, harnesses or corpus. The local working tree was compared to the remote tree, and unrelated repository entries were unchanged.
