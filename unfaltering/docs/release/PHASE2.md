# UNFALTERING 0.2.0-dev — Phase 2 implementation and validation

Date: 27 October 2026. **Development milestone; not public production-ready.**

Repository: https://github.com/RajChat-hub/R-
Branch: `feat/unfaltering-phase2-proxy-20261027`
Stacked draft PR: https://github.com/RajChat-hub/R-/pull/2
Base/foundation commit: `549c531e9ce2eff7fa5d5c64efc5d8fefce89c00`
Phase A gate: `dbf2a4f5f70a6c685ff9451bdeedf9b18798927c`
Validated Phase 2 code: `ebc4f69da145d33a89e1f2f5dde048c552d61a0e`

PR #2 targets the existing feature branch, not main. PR #1 remains unmerged. Existing jdk gitlink, archive pointer, script.rsharp and unrelated history were preserved. The original AGENTS.md, architecture, status, threat model, validation evidence and master directive were read before source edits.

## Delivered implementation

| Phase | Actual result |
|---|---|
| A — parser | Stateful bounded request decoding; headers/body/chunk states persist across reads; bounded sink and one-byte fragmentation checks; strict CL/TE rejection retained |
| A — reactor | Connection state and reactor declarations separated; bounded file/log workers; explicit memory admission, deadlines, readiness masks and work budgets |
| A — regression gate | GCC Debug and ASan/UBSan passed all 8 Phase A CTest groups before proxy integration; captured local logs and a passing remote checkpoint |
| B — transport | Original nonblocking epoll HTTP upstream client; fresh configured connection per transaction; no pooling, DNS resolution or retries |
| B — streaming | Fixed/chunked requests and fixed/chunked/EOF responses stream through individually bounded queues; message boundaries and backpressure enforced |
| B — policy | Hop-by-hop and Connection-nominated stripping; controlled Host; unverified Forwarded/X-Forwarded-*/X-Real-IP removed; optional generated direct-peer metadata |
| B — errors | Connect, send, upload, response, write and shutdown deadlines; 502/504/408 before commitment; truncate/close after commitment; disconnect/reset/early-EOF handling |
| C — config | Validated named numeric-IPv4 upstreams and root/proxy routes; old server.ufc retained; unknown/duplicate fields, invalid bounds and unresolved names fail before listening |
| C — examples | Shipped proxy.ufc and independent Python demonstration upstream; automated GET/POST/static example execution |
| D — validation | GCC Debug/Release and Clang sanitizer CI, static/hardening/proxy tests, request/config/response fuzz smoke campaigns, paired benchmark and measured storage |

The server and proxy remain native original C++20. Python is used only for independent tests, demonstrations and benchmark clients.

## Phase A gate and execution provenance

The persistent StreamParser replaced repeated whole-buffer request parsing. It retains bounded metadata and drains decoded payload rather than storing full request bodies. Exactly two IO workers share 64 total queued/active/completed credits; open/stat/traversal and 16 KiB pread run off the reactor. Jobs own descriptors and carry connection/epoch identities. Logging uses a separate fixed-capacity worker and nonblocking pipe sink; saturation drops records.

The original 22 static socket regressions passed after the refactor. Additional checks cover byte-at-a-time 64 KiB bodies, 4,000 tiny fragmented chunks, bounded sinks, the worker-credit limit, blocked logging and memory-based admission. Phase A Debug and ASan/UBSan command output is preserved in `evidence/phase2/phase-a-*.log`. Local leak detection was disabled because of the previously documented sandbox /proc restriction.

The local execution environment disconnected during Phase B. The proxy changes were restored through the connected repository, and the final native builds, socket tests, fuzz smoke tests and comparisons were executed on actual GitHub Linux runners. These are observed remote execution results, not inferred local successes.

## Final observed CI

Push: https://github.com/RajChat-hub/R-/actions/runs/40543528161
Stacked PR: https://github.com/RajChat-hub/R-/actions/runs/40543536247

Both runs passed. Push jobs: GCC 13.3 Debug, GCC 13.3 Release, Clang 18.1.3 Debug with ASan/UBSan and leak detection enabled, three libFuzzer smoke campaigns, and the comparison job. The comparison runs on push only by design; it is not a skipped failure on PR runs.

Each compiler test job passed **11/11 CTest groups**:

- 161 original unit assertions.
- Stateful decoder/sink/worker-budget unit checks.
- Response parser, framing, forwarding policy and Connection-option allocation-limit unit checks.
- 20,000 reproducible seeded HTTP/config mutations.
- CLI help/version and original example-config checks.
- 22 original real static-server network test methods.
- Three hardening integration methods.
- **30 native proxy integration methods**, with additional malformed-input subcases.
- Shipped proxy example/static/GET/POST demonstration.

Each HTTP request, configuration and upstream-response libFuzzer target completed **10,000 runs**. These are bounded seeded smoke campaigns, not exhaustive fuzzing.

Observed job/step results, compiler information, test outputs and sanitizer settings are in `evidence/phase2/github-ci.json`. Full command output remains in the linked GitHub job logs.

## Important test cases and fixes

Acceptance coverage includes GET/POST bodies, streaming before upload completion, chunked traffic, persistent pipelines that mix proxy/static responses, fragmented traffic, slow upload consumers, slow response readers, connect/response/send/upload timeouts, reset/early EOF, malformed response headers/chunks, client disconnect, half-close, forwarding trust, smuggling rejection, 8 MiB backpressure and 128 MiB streamed payload/hash checks, graceful shutdown with active proxy work, and old static-site defenses.

Adversarial testing exposed an asynchronous-file pipeline completion regression and a both-queues-full proxy wake-up deadlock; both were fixed. Final review added an exact 64 KiB write-budget completion regression and immediate transaction finalization, avoiding keep-alive stalls when no further readiness event is needed. The absolute proxy response deadline covers final downstream draining too.

Connection-option token count is bounded by header_count across the complete section, preventing comma-heavy metadata from allocating an arbitrarily large hop-header set within a header-byte limit. The current build explicitly requires 64-bit Linux.

## Memory and storage observations

The Release proxy suite streamed and verified **128 MiB** without whole-response buffering. Its native child-process peak across the suite was **46,952 KiB**, below the enforced noninstrumented **64 MiB** test ceiling. The largest checked combined proxy wire queues observed were **131,072 bytes**, within four configured 65,536-byte queue caps (262,144 bytes total). This diagnostic measures retained wire-string sizes at checks, not total process memory.

Instrumented child RSS is reported separately and is not judged against the noninstrumented ceiling because sanitizer quarantine/runtime overhead differs. Queue invariants remain enforced in sanitizer builds.

Measured project trees, including their builds:
- GCC Release full-test job: 4,020 KiB.
- GCC Debug full-test job: 27,476 KiB.
- Clang sanitizer full-test job: 36,744 KiB (about 35.9 MiB).
- Comparison job: current tree 2,148 KiB and separate baseline tree 1,320 KiB.

These measured project footprints are far below the 16 GiB working target. No massive third-party source/toolchain checkout or unrelated-file cleanup was used.

## Comparative development sample

`evidence/phase2/paired-benchmark.json` contains three matched pairs against the original foundation: GCC Release, 512 new-connection 1 KiB static responses, eight Python clients, warm local files, loopback, no TLS. The measured median-of-run throughput was about **5,139 requests/s for the foundation** and **5,172 for Phase 2** in that final sample.

An earlier paired sample is retained rather than discarded: it measured about 5,634 versus 5,304 requests/s. This run-to-run variation and the small client-limited workload do **not** establish a speed improvement or reliable regression estimate. Added worker isolation has CPU/context-switch costs. No Nginx comparison, commercial-capacity claim, idle-memory certification or sustained-load claim is made.

## Build and demonstration commands

From the repository root on supported 64-bit Linux:

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Release
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure --verbose
python3 unfaltering/examples/upstream.py
```

In another terminal:

```sh
./unfaltering/build/unfaltering --check --config unfaltering/examples/proxy.ufc
./unfaltering/build/unfaltering serve --config unfaltering/examples/proxy.ufc
```

Then:

```sh
curl -i http://127.0.0.1:8080/
curl -i http://127.0.0.1:8080/app/hello
curl -i -X POST --data 'native proxy body' http://127.0.0.1:8080/app/echo
```

The example's fixed localhost ports are for a local demonstration, not a deployed public service. Its automated smoke test uses temporary ports and verifies the same shipped files.

Sanitizers:

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++ -DUNFALTERING_SANITIZERS=ON
cmake --build unfaltering/build -j2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 UNFALTERING_INSTRUMENTED=ON ctest --test-dir unfaltering/build --output-on-failure --verbose
```

Use a fresh/reconfigured compiler build directory as appropriate; do not change compilers in an incompatible CMake cache. CI runs the compiler jobs in separate ephemeral runners.

## Remaining limits and release gates

Not implemented or not claimed: TLS/HTTPS upstreams, HTTP/2/3, tunnels/WebSockets, DNS resolution, pooling, cache, load balancing, FastCGI/PHP-FPM, multi-reactor distribution, transactional reload, WAF, IOCP/kqueue or general availability. Strict unsupported chunk extensions, nonempty trailers and Expect are rejected.

Configuration and filesystems are trusted operator inputs. Routes are not authorization controls. Logs are lossy operational diagnostics, not a durable audit trail. App admission budgets do not replace OS RSS/cgroup/socket limits. A permanently stuck kernel filesystem operation can delay worker joining beyond the network drain deadline; use trustworthy local storage and an external service-manager termination policy. A late client/body error cannot roll back application side effects; application idempotency remains necessary.

ThreadSanitizer, long-duration soak testing, expanded fuzz campaigns, exhaustive RFC conformance and independent security review are **not performed** release gates. First-party licensing remains pending. No public deployment, binary release or automatic merge occurred. Review the stacked draft and complete those gates before a separate TLS or release milestone.
