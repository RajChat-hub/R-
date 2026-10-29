# Native TLS operations — 0.3.0-dev

Development only; independent security review and release approval remain outstanding. The original C++20 reactor owns all sockets, epoll scheduling, deadlines and HTTP/proxy behavior. OpenSSL supplies cryptography/TLS only; it is not a web-server engine.

## Dependency policy

Build with a security-maintained OpenSSL 3.x installation, development headers and matching libssl/libcrypto. CMake requires the OpenSSL 3.0 API minimum and prints the detected version. It does not download, vendor or pin an obsolete TLS source release. Use a supported distribution's patched package and record its package revision, not just `openssl version`; distributors backport security fixes. For a separately managed upstream build, prefer a supported LTS line (3.5 support is scheduled through April 2030) and actively track security updates. A minimum API check cannot certify a package is patched. Ubuntu 24.04 CI upgrades the distribution's OpenSSL/libssl-dev before builds and records package versions. The local development runtime has OpenSSL 3.0.13 / Ubuntu 3.6 backport metadata; this is test provenance, not an endorsed production patch level.

OpenSSL 3 uses Apache-2.0. See THIRD_PARTY_NOTICES.md and third_party/OpenSSL-APACHE-2.0.txt. No project-wide first-party license grant is added. No FIPS certification, exhaustive library audit or app security certification is claimed. Operators own dependency monitoring, restart after updates and certificate renewal.

## Reproducible local HTTPS demonstration

From the repository root on 64-bit Linux, install CMake 3.20+, GCC/Clang, OpenSSL 3 development headers, Python 3 and the openssl/curl commands. For example the distro packages are `cmake g++ libssl-dev openssl python3 curl` on Ubuntu. No Python crypto package is required.

```sh
cmake -S unfaltering -B unfaltering/build -DCMAKE_BUILD_TYPE=Release
cmake --build unfaltering/build -j2
ctest --test-dir unfaltering/build --output-on-failure
python3 unfaltering/examples/make-test-certs.py unfaltering/examples/tls-local
./unfaltering/build/unfaltering --check --config unfaltering/examples/https.ufc
```

Terminal 1 (independent test application; not part of the native runtime):

```sh
python3 unfaltering/examples/upstream.py --port 9443 --certificate unfaltering/examples/tls-local/server.crt --private-key unfaltering/examples/tls-local/server.key
```

Terminal 2:

```sh
./unfaltering/build/unfaltering serve --config unfaltering/examples/https.ufc
```

Terminal 3, using actual CA and hostname verification:

```sh
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 https://localhost:8443/
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 https://localhost:8443/app/hello
curl --cacert unfaltering/examples/tls-local/ca.crt --resolve localhost:8443:127.0.0.1 --data 'hello TLS' https://localhost:8443/app/echo
curl --tlsv1.2 --tls-max 1.2 --cacert unfaltering/examples/tls-local/ca.crt https://localhost:8443/
curl --tlsv1.3 --tls-max 1.3 --cacert unfaltering/examples/tls-local/ca.crt https://localhost:8443/
```

Do not use `-k`. The generator creates a two-day **local test** CA and one-day leaf certificate plus an intentionally expired test leaf. It writes private keys only into the selected local directory, chmod 0600. It may overwrite test credentials there; do not point it at real certificate storage. Its outputs are ignored and never committed. Renew/regenerate expired test credentials. Public deployments need real certificates and separate operational approval. The shipped example is executed automatically with temporary ports/credentials in `tls_demo`.

## Configuration

The existing single `listen` endpoint can be plaintext (unchanged default) or TLS. This phase does not introduce simultaneous HTTP and HTTPS listeners in one process, SNI-based multi-certificate selection or live certificate reload. Existing virtual hosts share the configured certificate, which must cover their names.

```text
server {
 listen "127.0.0.1:8443";
 root "www";
 tls {
  certificate "certificates/fullchain.crt";
  private_key "certificates/server.key";
  handshake_ms 3000;
  shutdown_ms 500;
 }
 upstream application {
  address "127.0.0.1:9443";
  host "app.test";
  connect_ms 2000;
  response_ms 15000;
  tls {
   verify_name "app.test";
   sni "app.test";
   ca_file "certificates/internal-ca.crt";
  }
 }
 route prefix "/app" { proxy application; }
}
```

Paths resolve relative to the .ufc file. Unknown/duplicate options fail. Listener certificate and unencrypted PEM private key are mandatory together; malformed, mismatched, expired and not-yet-valid leaf certificates fail startup/`--check` before listening. File sizes are capped at 1 MiB. Encrypted-key prompting is deliberately disabled. Startup leaf checks do not establish that a public CA trusts the server chain; clients validate it. Intermediate validity, renewal and file permissions remain operator responsibilities. Use a private directory and key permissions 0600 under an unprivileged account; protect backup/config access.

Upstream addresses remain configured numeric IPv4:port; no DNS resolution or user-selected destination is introduced. `host` is HTTP policy, separate from TLS identity. `verify_name` is mandatory with upstream TLS. For DNS identities, explicit `sni` must exactly equal `verify_name`; SAN-based hostname verification disallows partial wildcards and CN-only fallback. For numeric IPv4 SAN verification, explicitly use `verify_name "127.0.0.1"; sni off;`. Empty/mismatched identity policies fail config parsing. `ca_file` is an optional explicit PEM trust store; omission uses maintained system trust. There is no insecure-verification switch. Invalid CA files fail before listening; actual peer expiration, chain/name/SNI failures are checked at handshake and become 502. Handshake/connect deadline expiry becomes 504.

Listener handshake_ms defaults to 3000 (100–60000); shutdown_ms defaults to 500 (100–5000). Listener TLS shutdown deadline is separate from the server network drain deadline. Upstream connect_ms includes both TCP connect and TLS handshake; response_ms remains an absolute transaction deadline. Existing read/write/upload limits apply to TLS application traffic. Startup certificate/trust loading is synchronous before external traffic, not performed per request on the reactor.

## Protocol and scheduling policy

TLS 1.2/1.3 only, OpenSSL security level 2, ECDHE AEAD TLS 1.2 suites and the standard AES-GCM/ChaCha20 TLS 1.3 suites. Compression, renegotiation, session cache/tickets and early data are disabled. HTTP/1.1 is the only ALPN selection; an offered ALPN list without it fails. Absence of ALPN permits HTTP/1.1. OpenSSL's peer certificate-list limit is 64 KiB and verification depth is 8.

Each endpoint has its own SSL object used only on the reactor thread. SSL_get_error is evaluated immediately on that thread with a cleared OpenSSL error queue. Read and write WANT directions alter epoll interests, including cross-direction readiness. Pending write retries own a stable maximum-16-KiB snapshot and retain the original length; application queue compaction cannot change the bytes being retried. Reads use a fixed 8-KiB staging buffer. Buffered decrypted data is pumped on the bounded reactor sweep even without a new socket edge. This may add up to the normal 50-ms sweep interval under backpressure. Read/write network budgets and original bounded queues remain active.

A pending OpenSSL write is retried before another TLS operation; a peer that refuses to read while trying to send an early response can reach the configured transaction deadline. There is no promise of arbitrary bidirectional application tunnelling. HTTP upgrades/tunnels remain unsupported. The server never retries an application operation automatically.

Downstream completed/idle connections exchange close_notify with a bounded close deadline, then close even if a peer does not cooperate. Errors, partial messages and drain deadline expiry can force socket closure. Completed upstream transactions make a best-effort nonblocking close_notify attempt and close their fresh, never-pooled socket without waiting for reciprocal shutdown. No connection reuse ambiguity is introduced.

Authenticated bytes already buffered at a terminal upstream transport error are first passed through the HTTP parser. An exact fixed/chunked message can finish; incomplete messages and close-delimited bodies without verified TLS close_notify fail. Before HTTP commitment, failures yield gateway errors; after commitment the response truncates/closes, without a fake final chunk or second response. Application side effects cannot be rolled back.

## Budgets, logs and shutdown

When either listener or any configured upstream uses TLS, add **512 KiB per admitted downstream connection** to the previous reservation. This allows 256 KiB per potential TLS endpoint for SSL state, bounded staging buffers/records and certificate processing. It is conservative admission accounting, **not an OpenSSL allocator cap or hard RSS limit**. Fixed contexts/trust stores, allocator retention, stacks, worker jobs and kernel socket memory need additional headroom and OS/cgroup enforcement. Large trusted configurations with many upstream trust stores increase startup memory independently.

The representative 32-connection TLS acceptance fixture uses a 64-MiB application admission budget, a separate **192-MiB noninstrumented process-RSS test ceiling**, and a **768-MiB instrumented ceiling**. These are test guardrails for this fixture, not universal promises. The 192-MiB ceiling leaves headroom above 32 conservative TLS reservations plus original queues, contexts, stacks and allocator overhead; actual measured high-water values are recorded in PHASE3.md. Owned client/upstream network endpoints are counted and bounded by twice max_connections; worker and fixed descriptors need additional OS allowance. A real RLIMIT_NOFILE=32 failure/recovery test exercises TLS descriptor exhaustion. Plaintext regression tests retain their original limits.

Structured `tls_error` diagnostics contain only upstream/downstream direction and numeric library/verification codes. They exclude peer certificate text, key material, bodies, credentials, query strings and paths. `proxy_failure` records status, decoded-byte and queue counts. Logs are fixed-capacity, lossy operational diagnostics, never a durable audit trail. Missing diagnostics do not establish success. Use external monitoring and a service manager.

Shutdown cancels queued filesystem work and rejects new jobs; active workers remain joined. C++ cannot safely terminate a thread stuck in an uninterruptible filesystem syscall. Regular-file logging has the same storage dependency. Trust local storage and use an external process-level termination policy; no hard real-time process shutdown guarantee is made.

## Reference material

- OpenSSL nonblocking guide: https://docs.openssl.org/3.5/man7/ossl-guide-tls-client-non-block/
- Read/write retry contracts: https://docs.openssl.org/3.5/man3/SSL_write/ and https://docs.openssl.org/3.5/man3/SSL_get_error/
- Verification: https://docs.openssl.org/3.5/man3/SSL_set1_host/
- Release maintenance: https://openssl-library.org/policies/releasestrat/
- Licensing: https://github.com/openssl/openssl/blob/master/LICENSE.txt
