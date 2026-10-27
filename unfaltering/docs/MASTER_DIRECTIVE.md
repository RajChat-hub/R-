# CODEX MASTER ENGINEERING DIRECTIVE
## UNFALTERING™ — Original Commercial C++ Web Server and Reverse Proxy

**Product:** UNFALTERING™ Web Server  
**Tagline:** Built to Endure. Engineered to Serve.  
**Publisher:** SalesAutoMate™  
**Repository:** https://github.com/RajChat-hub/R-  
**Primary language:** ISO C++20  
**Development environment:** Codex Cloud, approximately 20 GB total workspace storage budget  
**Project:** Original standalone commercial-grade hybrid HTTP web server, reverse proxy, and extensible networking platform.

---

# 1. YOUR MISSION

Act as the principal C++ systems architect, networking engineer, security engineer, performance engineer, and release engineer responsible for implementing UNFALTERING.

This is a REAL SOFTWARE ENGINEERING AND IMPLEMENTATION TASK, not merely a request for documentation, pseudocode, architectural diagrams, or a proof-of-concept landing page.

Design, implement, compile, run, test, debug, and progressively develop an original, independently operated C++ web-server product.

UNFALTERING is intended to mature into a serious commercial alternative to Apache HTTP Server and Nginx.

Its architecture must combine:

1. Native HTTP web serving.
2. Asynchronous, event-driven networking.
3. Reverse proxying.
4. Multiple websites and virtual hosts.
5. Application integration.
6. TLS termination.
7. Traffic routing and load balancing.
8. Security and abuse prevention.
9. Caching.
10. Metrics and operational diagnostics.
11. Safe configuration and reloads.
12. Cross-platform portability.
13. Extensible modular architecture.
14. Professional installation and operation.
15. Long-term maintainability and commercial distribution.

Implement real functionality incrementally, with rigorous acceptance gates. Do not pretend the entire feature set is production-ready merely because a development skeleton exists.

# 2. NON-NEGOTIABLE IDENTITY AND ORIGINALITY

The following decisions are FINAL:

- Product name: UNFALTERING.
- Publisher: SalesAutoMate.
- Primary implementation language: C++20.
- Core architecture: original hybrid web server and reverse proxy.
- Core networking: original implementation.
- HTTP/1.1 parsing and message processing: original implementation.
- Connection management and scheduling: original implementation.
- Request routing, configuration, and proxy orchestration: original implementation.

DO NOT replace the server with Apache, Nginx, Caddy, Envoy, HAProxy, or another existing web-server engine.

DO NOT create a branded wrapper that secretly depends on an existing web server to perform all essential operations.

DO NOT use Boost.Asio, libuv, or another prebuilt event-loop implementation as the core networking engine.

Implement appropriate operating-system event facilities directly:

- Linux: epoll.
- macOS/BSD: kqueue.
- Windows: IOCP.
- Use a clearly documented abstraction layer and develop platform backends incrementally.

The server itself must be substantively original SalesAutoMate C++ code.

Use established, appropriately licensed implementations of cryptographic algorithms and TLS rather than attempting to invent new cryptography.

OpenSSL or another mature TLS library may be integrated after reviewing compatibility, license obligations, updates, and security maintenance.

Do not copy third-party source code, proprietary software, or existing web-server implementations without authorization. Record all dependencies and licenses.

# 3. REPOSITORY SAFETY

Inspect the existing repository before making changes.

The designated repository is:

https://github.com/RajChat-hub/R-

Its default branch is currently main, and it is publicly accessible.

The repository has existing unrelated content, including:

- jdk
- script.rsharp
- openjdk-24-ea+19_linux-aarch64_bin.tar.gz

DO NOT assume it is empty.

DO NOT delete, overwrite, rewrite, or damage existing files or Git history.

Create the independent product under:

`unfaltering/`

Keep source code, tests, documentation, build configurations, deployment examples, benchmarks, and packaging within that project directory wherever practical.

GitHub Actions workflows may be created at the repository's `.github/workflows/` location if necessary, but preserve any pre-existing workflows.

Work on a dedicated feature branch and open a pull request when possible.

Do not force-push or directly alter unrelated products.

Never commit credentials, private keys, certificates containing private keys, tokens, deployment secrets, or generated build caches.

# 4. PLATFORM AND TOOLCHAIN

Use:

- C++20.
- CMake and CTest.
- GCC and Clang on Linux.
- MSVC on Windows when the Windows backend is implemented.
- Apple Clang on macOS when supported.
- GitHub Actions for automated validation.
- Python standard library or shell scripts for test orchestration when useful.
- C++ standard library for ordinary utilities.

Prioritize Linux x86-64 as the first executable production-oriented target.

Design portable interfaces immediately, but do not label Windows or macOS operationally supported until their respective implementations compile and pass platform tests.

Avoid heavyweight dependencies unless strongly justified.

Keep the networking runtime entirely in native C++.

# 5. FUNDAMENTAL ARCHITECTURE

Create clearly separated modules.

## 5.1 Core Runtime

Responsibilities:

- Process initialization.
- Startup validation.
- Server lifecycle.
- Thread and worker management.
- Graceful shutdown.
- Error reporting.
- Signal handling where supported.
- Bounded resource allocation.
- Runtime health reporting.

Apply RAII throughout C++ code.

Use ownership-safe primitives such as unique_ptr, scoped handles, standard containers, spans, and suitable atomic operations.

Avoid unsafe global mutable state.

## 5.2 Event Engine

Implement an original nonblocking event-loop abstraction supporting:

- Listener creation.
- Socket registration.
- Readiness notifications.
- Partial reads and writes.
- Connection registration/removal.
- Backpressure.
- Idle timeouts.
- Connection limits.
- Graceful connection closure.
- Multithreaded worker distribution.

Use epoll for the initial Linux implementation.

Correctly handle EAGAIN, EINTR, half-closed connections, descriptor reuse, resource exhaustion, and shutdown races.

Do not busy-spin under idle or blocked conditions.

## 5.3 HTTP/1.1 Engine

Implement an original standards-conscious HTTP/1.1 request parser, response serializer, and connection state machine.

Use RFC 9110 and RFC 9112 as core references.

Support:

- Request line parsing.
- Header parsing.
- Host validation.
- HTTP/1.1 persistent connections.
- Request-body framing.
- Content-Length.
- Chunked transfer coding.
- Safe request-body limits.
- Safe header size and count limits.
- Proper status codes.
- Correct response framing.
- GET and HEAD handling.
- Protocol error responses.
- Configurable timeouts.
- Incremental parsing across partial network reads.

Handle ambiguous message framing defensively.

Explicitly reject inconsistent Content-Length, unsafe Transfer-Encoding/Content-Length combinations, malformed chunk framing, prohibited control characters, and invalid HTTP syntax.

Avoid request smuggling vulnerabilities, parser differentials, memory corruption, and response splitting.

Implement tests for fragmented packets, malformed headers, body framing, and connection reuse.

## 5.4 Static Content Engine

Implement native static-content serving with:

- Configurable document roots.
- Virtual hosts.
- MIME detection.
- GET/HEAD.
- Efficient buffered transmission.
- Optional platform-supported zero-copy operations.
- Correct Content-Length.
- Cache-control configuration.
- Directory index handling.
- Safe 404/403 responses.
- Secure pathname handling.

Prevent directory traversal, encoded traversal, null-byte injection, path normalization mistakes, and symlink escapes.

Use safe filesystem access strategies rather than relying on string-prefix comparisons.

Do not expose arbitrary filesystem paths.

Add conditional GET, byte ranges, compression, and advanced static optimization through separately tested milestones.

## 5.5 Routing Engine

Implement:

- Host-based routing.
- Exact path matching.
- Prefix routing.
- Route priority.
- Method restrictions.
- Configurable default hosts.
- Static routes.
- Proxy routes.
- Consistent error behavior.

Routing must be deterministic and testable.

## 5.6 Reverse Proxy Engine

Build the original C++ reverse proxy transport and request/response forwarding logic.

Support:

- Configured HTTP upstreams.
- Connection establishment and reuse where safely implemented.
- Hop-by-hop header filtering.
- Correct HTTP framing.
- Streaming request and response bodies.
- Backpressure.
- Connect/read/write timeouts.
- Error propagation.
- Upstream failures.
- Original client metadata using explicit trust rules.
- Safe forwarding of allowed headers.

Do not trust arbitrary client-supplied Forwarded or X-Forwarded-* headers.

Never silently retry non-idempotent requests.

Implement loopback-based integration tests against a small independent HTTP test upstream.

## 5.7 TLS Engine

Integrate a maintained, audited TLS library.

Implement:

- HTTPS listeners.
- Certificate/key configuration.
- TLS 1.2/1.3 where supported by the selected library and policy.
- Secure protocol defaults.
- SNI-based certificate selection when developed.
- Graceful TLS shutdown.
- Certificate validation for HTTPS upstreams.
- Safe handling of key files.

Do not store private keys in the repository.

Do not invent encryption algorithms.

Certificate automation and HTTP/2-over-TLS can be added after the core TLS layer passes security and integration tests.

## 5.8 Load Balancing

Implement progressively:

- Round-robin.
- Least-connections or another justified algorithm.
- Passive upstream-failure detection.
- Active health checks.
- Failover.
- Configurable pool membership.
- Retry policies restricted to safe cases.
- Connection and request limits per upstream.

Test backend failure, recovery, uneven backend speeds, and graceful shutdown.

## 5.9 Caching

Implement an original bounded cache subsystem.

Start with an explicitly configured, memory-bounded cache and add disk caching later.

Requirements:

- Correct HTTP cache semantics.
- Expiration.
- Bounded memory consumption.
- Eviction policy.
- Cache statistics.
- Cache-key correctness.
- Safe handling of authorization and cookies.
- Cache bypass rules.

Never cache confidential, personalized, or authenticated responses without explicit safe policy.

## 5.10 Security Engine

Implement a configurable security policy layer for:

- Request size limits.
- Header size/count limits.
- Connection limits.
- IP-based rate limiting.
- Slow-client protection.
- Request timeouts.
- Allowed HTTP methods.
- Optional host allowlists.
- Safe response headers.
- Administration endpoint restrictions.
- Secure logging.

Every resource-control mechanism must have tests proving limits are enforced.

Do not advertise this as a complete WAF unless a genuinely supported and tested WAF capability exists.

## 5.11 Configuration Engine

Design an original declarative configuration format, provisionally using the `.ufc` extension.

Features:

- Strict syntax.
- Explicit server/listener blocks.
- Virtual hosts.
- Route rules.
- Static roots.
- Reverse proxy upstreams.
- TLS configuration.
- Security policies.
- Cache configuration.
- Logging and metrics settings.
- Human-readable parser errors.
- Schema validation.
- Unknown-key rejection.
- Duplicate/conflicting listener detection.

Provide:

`unfaltering --check --config path/to/server.ufc`

and an actual startup command such as:

`unfaltering serve --config path/to/server.ufc`

Do not invent CLI flags in documentation unless they are implemented.

Safe live configuration reload should be developed with transactional validation, atomic activation, and rollback behavior.

A failed reload must not destroy the running configuration.

## 5.12 Observability

Implement:

- Structured access logs.
- Structured error logs.
- Request identifiers.
- Request counts.
- Active connection counts.
- Upstream health metrics.
- Latency measurements.
- Resource consumption metrics where supported.
- Optional Prometheus-compatible metrics endpoint.

Metrics and administration endpoints must default to loopback or other appropriately protected access.

Sensitive tokens, credentials, and request bodies must not be leaked to logs.

# 6. PHP AND APPLICATION INTEGRATION

UNFALTERING must eventually support common commercial web applications.

Create a native FastCGI client subsystem for PHP-FPM rather than depending on Apache or Nginx for PHP handling.

The FastCGI implementation must correctly handle records, parameters, stream handling, errors, process termination, and security-sensitive configuration.

Provide a future production-oriented example for Laravel using PHP-FPM.

For Node.js, Python, Java, and other application runtimes, use the original HTTP reverse proxy.

Do not embed entire language runtimes unnecessarily.

Clearly distinguish:

- Natively served static content.
- FastCGI-connected PHP applications.
- Reverse-proxied HTTP applications.

# 7. LONG-TERM HTTP/2 AND HTTP/3 ROADMAP

Design an extensible protocol interface that can later accommodate HTTP/2 and HTTP/3.

Do not pretend HTTP/2 or HTTP/3 is implemented by accepting connections and returning HTTP/1.1 responses.

When implementing these protocols, follow the relevant specifications and provide exhaustive protocol tests.

Never ship an incomplete or insecure protocol parser enabled by default.

Focus first on correctness and security of HTTP/1.1.

# 8. RESOURCE MANAGEMENT — 20 GB WORKSPACE

The Codex environment has a practical storage budget of approximately 20 GB.

Inspect actual available disk space and memory before building.

Target a peak workspace footprint below 16 GB, leaving headroom for operating-system and tool requirements.

Use incremental CMake builds.

Do not clone entire large third-party repositories unless essential.

Do not download unnecessary toolchains, operating-system images, datasets, or container images.

Avoid retaining multiple complete debug/release build directories simultaneously.

Use narrowly scoped dependency downloads and pin versions.

Keep generated artifacts and compiler outputs outside Git tracking.

Use `.gitignore` for build outputs.

Configure appropriate cache limits.

Never delete unrelated user files to recover storage.

If disk capacity becomes constrained, prune only verified project-generated temporary artifacts.

Document measured storage use.

# 9. SECURITY, RELIABILITY, AND CODE QUALITY

Treat all network inputs as hostile.

Adopt:

- RAII resource ownership.
- Defined maximum request sizes.
- Integer overflow checks.
- Safe buffer boundaries.
- Explicit connection state machines.
- No unbounded queues.
- No uncontrolled memory growth.
- No unbounded thread spawning.
- Fail-closed security policies where appropriate.
- Deterministic cleanup paths.
- No uncaught exceptions escaping event-loop boundaries.
- No disabled TLS certificate validation in production.
- No hard-coded credentials.

Add strict compiler warnings.

Use AddressSanitizer and UndefinedBehaviorSanitizer in supported test builds.

Use ThreadSanitizer where feasible and separately tested.

Add fuzzing harnesses for HTTP parsing and configuration parsing.

Test malformed requests, slow clients, connection resets, upstream failures, oversized payloads, partial reads/writes, and concurrent shutdown.

Write a threat model covering trust boundaries and exposed interfaces.

Do not market software as secure merely because it compiles.

# 10. BUILD A PROFESSIONAL PRODUCT

Create a real product structure, not a single giant C++ file.

Expected layout:

unfaltering/

- CMakeLists.txt
- README.md
- AGENTS.md
- cmake/
- include/unfaltering/
- src/core/
- src/platform/
- src/net/
- src/http/
- src/routing/
- src/static/
- src/proxy/
- src/tls/
- src/security/
- src/cache/
- src/config/
- src/metrics/
- src/fastcgi/
- src/cli/
- tests/unit/
- tests/integration/
- tests/fuzz/
- benchmarks/
- examples/
- packaging/
- docs/architecture/
- docs/security/
- docs/protocol/
- docs/operations/
- docs/roadmap/
- docs/release/
- THIRD_PARTY_NOTICES.md
- CHANGELOG.md

Adapt this structure only when a better technical organization is justified.

Do not create empty modules merely to make the project look complete.

Maintain explicit feature status for every module.

# 11. COMMERCIAL IDENTITY AND LICENSING

Use product branding consistently:

UNFALTERING™ Web Server  
A SalesAutoMate™ Technology  
Built to Endure. Engineered to Serve.

The intended product is commercially distributable.

Do not assume that making a GitHub repository public grants everyone an open-source license.

Do not add MIT, GPL, Apache-2.0, or another project-wide software license without explicit owner approval.

Document that the first-party licensing decision is pending and third-party licenses must be honored.

Include appropriate copyright notices for original first-party code.

Maintain a third-party dependency and license inventory.

Do not claim legal ownership of third-party libraries.

Do not include proprietary third-party source unless expressly licensed.

# 12. EXECUTION PLAN — IMPLEMENT, DO NOT JUST PLAN

Develop through verified milestones.

## Milestone 0 — Repository audit and foundation

Inspect repository contents and available system resources.

Create the independent project directory without damaging existing files.

Implement:

- CMake project.
- Build scripts.
- Initial source organization.
- Compiler warning policy.
- Unit-test framework.
- Initial documentation.
- Architecture decisions.
- Git ignore rules.
- CI foundation.

Build a real executable that supports `--version` and `--help`.

Acceptance: executable compiles, runs, prints accurate product information, and is covered by basic tests.

## Milestone 1 — Functional HTTP/1.1 server

Implement the native Linux event loop, TCP listener, connection state machine, HTTP parser, response serializer, and static file serving.

Acceptance:

- Starts on configured loopback port.
- Serves a real HTML file.
- Handles GET and HEAD.
- Returns proper status codes.
- Processes fragmented requests correctly.
- Supports persistent connections safely.
- Rejects invalid framing.
- Enforces size/time limits.
- Stops gracefully.
- Passes automated tests.

This is the FIRST critical working deliverable.

## Milestone 2 — Real configuration and routing

Implement the original configuration parser, configuration validation, virtual hosts, and route handling.

Acceptance: two local hostnames can route to different static document roots using the implemented configuration format.

## Milestone 3 — Reverse proxy

Implement HTTP upstream forwarding, streaming, timeout policy, hop-by-hop header handling, and backpressure.

Acceptance: run a real local upstream test application behind UNFALTERING, verify ordinary and error responses, test slow upstreams and connection resets, and pass integration tests.

## Milestone 4 — TLS and operational hardening

Integrate audited TLS, add HTTPS, certificate configuration, safer defaults, logging, and shutdown/recovery testing.

Acceptance: HTTPS works with test certificates, TLS integration tests pass, and certificate validation is enabled for HTTPS upstreams.

## Milestone 5 — Load balancing and metrics

Implement multiple upstreams, health checks, load balancing, and operational metrics.

Acceptance: automated tests verify routing across healthy backends and appropriate behavior when backends fail or recover.

## Milestone 6 — Cache and PHP-FPM

Implement bounded caching and the original FastCGI gateway, with an actual PHP-FPM integration test where the environment supports it.

Acceptance: real PHP request execution through PHP-FPM, correct HTTP responses, cache behavior tests, and security checks.

## Milestone 7 — Commercial distribution preparation

Prepare:

- Linux packaging.
- Installation scripts.
- Service management example.
- Configuration reference.
- Upgrade procedure.
- Backup/recovery guidance.
- Security documentation.
- Operational runbook.
- Build and release CI.
- Versioned release artifacts.

Expand Windows and macOS support only when those backends are actually implemented and tested.

Do not declare general availability without all required release gates.

# 13. TESTING REQUIREMENTS

Create actual executable tests, including:

1. Unit tests for parsing and serialization.
2. HTTP protocol compliance tests.
3. Malformed-input regression tests.
4. Network fragmentation tests.
5. File-serving and path-security tests.
6. Configuration parser tests.
7. Reverse proxy integration tests.
8. Fault-injection tests.
9. Graceful shutdown tests.
10. Memory and sanitizer tests.
11. Concurrency and stress tests.
12. Representative performance benchmarks.

Provide executable test commands.

Run every relevant test after changing affected components.

Do not replace failed tests with mocked success messages.

Do not silently skip failing tests.

Distinguish clearly between passed, failed, skipped, and not-run checks.

# 14. PERFORMANCE AND BENCHMARK POLICY

Evaluate UNFALTERING against Apache, Nginx, and optionally Caddy when matching benchmark environments are available.

Measure:

- Requests per second.
- Median latency.
- p95 and p99 latency.
- CPU usage.
- Peak memory.
- Idle memory.
- Connection throughput.
- File-serving throughput.
- Error rate.
- Graceful restart/reload behavior.

Use matched payloads, client counts, network conditions, CPU allocation, cache state, and TLS settings.

Record hardware, operating system, compiler flags, exact server versions, benchmark tool, and commands.

Never manufacture performance numbers.

Never claim to outperform Apache or Nginx without reproducible supporting measurements.

# 15. GITHUB AND CONTINUOUS INTEGRATION

Use a feature branch.

Commit coherent development milestones.

Create or update GitHub Actions workflows for:

- Linux Debug build.
- Linux Release build.
- CTest.
- Compiler warnings.
- AddressSanitizer/UndefinedBehaviorSanitizer tests when supported.
- Static analysis where practical.
- Dependency/security review where available.

Add platform build jobs as platform support becomes real.

Use minimal GitHub Actions permissions.

Pin third-party actions to vetted versions or commit hashes where practical.

Avoid publishing binary releases automatically until packaging and security gates are met.

Open a pull request with complete implementation details and verified test results when the environment supports it.

# 16. DOCUMENTATION REQUIREMENTS

Maintain real documentation describing the actual code.

Include:

- Architecture overview.
- Module boundaries.
- Build instructions.
- Developer instructions.
- Runtime commands.
- Configuration specification.
- Deployment examples.
- Limitations.
- Security threat model.
- Dependency/license inventory.
- Feature status matrix.
- Benchmark methodology.
- Release-readiness checklist.
- Known issues and future milestones.

Do not mark unimplemented features as complete.

Documentation examples must match actual implemented command-line options and configuration syntax.

# 17. EXECUTION DISCIPLINE

Important: perform actual development in this task.

Start immediately by examining the repository and engineering environment.

Then execute Milestone 0 and Milestone 1 before attempting advanced features.

Continue into later milestones when sufficient execution time, resources, and validated foundations are available.

Do not spend the entire session producing only planning documents.

Do not sacrifice correctness or tests to create the appearance of a complete product.

When encountering failures:

1. Inspect the error.
2. Determine the underlying cause.
3. Fix the implementation.
4. Rebuild.
5. Rerun relevant tests.
6. Document remaining limitations.

Do not claim successful execution without evidence.

Do not invent successful builds, performance results, screenshots, CI checks, or deployment URLs.

If a milestone cannot be completed in the current run, commit the working progress and record exactly what remains.

Keep the repository in a buildable state wherever possible.

# 18. FINAL REPORT AND ACCEPTANCE

At the end of the Codex run, report:

- Repository and working branch.
- Commit hash.
- Pull request URL, if created.
- Actual completed milestones.
- Implemented features.
- Unimplemented features.
- Build commands executed.
- Test commands and results.
- Security checks completed.
- Known vulnerabilities or limitations.
- Measured workspace disk usage.
- Platform support status.
- Exact steps required for the next milestone.

Include a real example showing how to start the executable and serve a test page.

If a feature is not actually working, explicitly label it incomplete.

Do not call UNFALTERING production-ready until the defined release criteria, operational validation, and appropriate security review are complete.

# FINAL DIRECTIVE

Build UNFALTERING as a genuine, independent, professional C++ server technology.

Prioritize originality, correctness, secure network handling, resource efficiency, extensibility, operational reliability, and maintainability.

Produce working software, not a simulation.

Maintain the long-term commercial architecture, but implement incrementally with verifiable quality gates.

Begin with the repository audit, establish the build system, implement the original HTTP/1.1 server, and continue through the engineering milestones as far as validated execution permits.

**UNFALTERING™ — Built to Endure. Engineered to Serve.**

