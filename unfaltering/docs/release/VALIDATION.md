> Historical Phase 1 evidence. Current Phase 2 implementation, tests and limitations are in [PHASE2.md](PHASE2.md).

# Validation — 27 October 2026

## Executed locally

Linux x86-64, kernel 6.18.55, glibc 2.39, GCC 13.3.0, CMake/CTest 3.31.6, Python 3.12.14. CMake was installed in a temporary tool directory, not vendored. clang-format 18.1.8 was used for C++ formatting. Nine CPUs were visible to the container; no claim of dedicated CPU allocation is made.

The original directive was read fully and preserved in `docs/MASTER_DIRECTIVE.md`. Audited base commit: `6e3b107314c7c67681432db1bd29bf7568eb10e8`. No pre-existing AGENTS.md or workflow was present in the observed tree. Existing entries preserved:

| Path | Original object |
|---|---|
| jdk (gitlink) | 41ee582df8c65f2f26b21e46784cf0bc4ece0585 |
| openjdk-24-ea+19_linux-aarch64_bin.tar.gz (LFS pointer) | 3ff005174ca2c0349f2de3b34a12d4dedd85cdae |
| script.rsharp | 70fdbbc5b9497826db0e7cb1d66cc67a5660cf8d |

No submodules or large JDK archive were downloaded. Work is isolated under `unfaltering/` plus `.github/workflows/unfaltering-ci.yml` on `feat/unfaltering-http-foundation-20261027`.

Both final configurations passed **6/6 CTest groups**: 161 unit assertions, 20,000 reproducible seeded HTTP/config mutations, CLI version/help, example configuration, and 22 real socket integration test methods (with additional subcases). The mutation test is not coverage-guided fuzzing. Tests do not rely on C++ `assert`, so Release retains validation.

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Debug -DUNFALTERING_SANITIZERS=ON
cmake --build unfaltering/build -j2
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir unfaltering/build --output-on-failure
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Release -DUNFALTERING_SANITIZERS=OFF
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure
```

The actual executable used for cmake/ctest locally was `/tmp/unfaltering-tools/cmake/data/bin/`; commands above are the equivalent PATH-based commands. One build directory was reused. Compiler policy: C++20, -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror. Release uses CMake's GCC -O3 -DNDEBUG defaults; sanitizer Debug uses ASan/UBSan, frame pointers and non-PIE linking in this environment. Captured final test logs are in `evidence/`.

Socket cases cover GET/HEAD, HEAD errors without bodies, two different host roots, exact/prefix routing, fragmented bytes, fixed/chunked body boundaries, pipelining, maximum requests, duplicate/ambiguous framing, header/body limits, traversal/encoded traversal/symlink escapes, full and half-close behavior, large partial writes, request/body/idle/write deadlines, connection limits, 120 concurrent/reset client attempts, descriptor exhaustion under RLIMIT_NOFILE=24, recovery, and graceful shutdown with blocked output. A real half-close/backpressure truncation defect was reproduced with a failing test and fixed by excluding RDHUP from write-state interest. The regression now passes.

## Verified remote CI

Both GitHub push run 40488459793 and PR run 40488465032 passed at code commit `04c7a9077f5a894f221c47565cacad803bb08ff8`. All four jobs passed: GCC Debug, GCC Release, Clang 18.1.3 Debug with ASan/UBSan and leak detection enabled, and libFuzzer smoke checks. Each compiler test job ran all six CTest groups successfully. HTTP and config fuzzers each completed 10,000 seeded runs. These are bounded smoke campaigns, not exhaustive fuzzing. Observed job/step results and compiler/test excerpts are preserved in `evidence/github-ci.json`; full logs remain at the linked workflow runs. Later evidence-documentation commits do not change the tested C++ source.

https://github.com/RajChat-hub/R-/actions/runs/40488459793

https://github.com/RajChat-hub/R-/actions/runs/40488465032

## Limits / checks not passed or not run

- LeakSanitizer was attempted and failed at its environment inspection stage: child task entries under `/proc` are inaccessible in this sandbox. It did not produce a successful leak report. ASan/UBSan were rerun with only leak detection disabled and passed. Remote Clang CI subsequently passed with leak detection enabled. This does not remove the local sandbox limitation.
- Local Clang compilation and libFuzzer execution: NOT RUN. Clang was absent and the system package installer could not write its cache. No false fallback compiler result is claimed. Remote Clang builds and both libFuzzer smoke campaigns subsequently passed as recorded above.
- TSan, long-duration soak tests, independent security audit and exhaustive RFC conformance: NOT RUN. Runtime is presently single-threaded.
- Apache/Nginx/Caddy comparative benchmarks: NOT RUN. No performance superiority claim is made.
- Windows/MSVC, macOS/Apple Clang, IOCP and kqueue: NOT IMPLEMENTED / NOT TESTED.
- Initial remote CI failed before compilation because the existing jdk gitlink lacks .gitmodules metadata and checkout auth cleanup invoked submodule foreach. Sparse checkout alone still left the gitlink in the index and did not fix submodule foreach. The workflow now uses public read-only Git fetch/checkout without the action's submodule/auth-cleanup calls, selecting only unfaltering/ and .github/ in the worktree and preserving unrelated repository entries. Actual rerun results must be read from GitHub; workflow definitions alone do not establish passing checks.

## Development benchmark

Executed `python3 unfaltering/benchmarks/loopback.py unfaltering/build/unfaltering` against the final GCC Release executable. The JSON in `evidence/loopback-release.json` records its binary hash, environment and measured results: 512 successful 1 KiB responses, eight Python clients, loopback, warm file, new connection per request, no TLS. This is a very short development sample and includes Python-client overhead; it is not a capacity, idle-memory, sustained-throughput or commercial-performance certification. The child peak RSS includes process creation and is not a separately measured steady-state server RSS.

## Storage and release

Measured project tree including the single Release build was approximately 2.9 MiB at the code-build checkpoint; temporary CMake tools 78 MiB and formatter 3.8 MiB. This is far below the directive's 16 GiB working target. Final report may include a later exact tree measurement as documents are added.

No binary release, deployment, main-branch merge or first-party licence grant has occurred. Remaining capabilities and release gates are recorded in STATUS.md and the threat model. The code milestone has passing remote CI; commercial readiness and future features remain separate gates. Review the draft PR before advancing development.
