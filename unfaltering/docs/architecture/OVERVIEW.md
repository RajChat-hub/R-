# Architecture and current boundaries

`src/cli` validates the CLI, loads configuration and starts the runtime. `src/config` tokenizes the strict bounded configuration and resolves roots relative to its file. `src/http` owns the original message parser and response serializer. `src/routing` selects exact routes before longest segment-boundary prefixes. `src/static` performs descriptor-relative file traversal. `src/net` owns the Linux reactor and connection lifecycle. `Fd` closes descriptors on all ordinary and exceptional ownership paths.

The initial runtime is a single-threaded, level-triggered epoll reactor. Listener and signalfd IDs are reserved; client events use monotonically increasing connection identities rather than descriptor numbers. Stale readiness events cannot address a newer connection reusing an old descriptor. SIGINT/SIGTERM enter through signalfd; there is no signal-handler access to C++ containers.

One connection alternates between reading one bounded request and writing one response. While writing, EPOLLIN is disabled. Already received pipeline bytes are retained and dispatched only after the previous response. Reading and writing have 64 KiB event budgets. File payloads use a 16 KiB buffer and pread rather than loading the file. The kernel TCP buffers supply additional bounded buffering. A held reserve descriptor and temporary listener deregistration handle EMFILE/ENFILE without a readiness hot loop.

HTTP parsing accepts partial buffers and returns incomplete/complete/error plus exact consumed bytes. It reparses the bounded current buffer on new data, rather than maintaining an incremental lexer cursor; worst-case fragmented chunk performance needs dedicated fuzz/performance hardening. Headers/body/framing bytes have explicit limits. Chunk extensions and nonempty trailers fail closed. GET/HEAD may carry framed bodies, which are consumed before reuse; static methods other than these return 405.

The filesystem root is opened once. Each path component is opened with openat and O_NOFOLLOW; intermediate components must be directories, final content must be a regular file, and directory indexes are opened relative to a held directory descriptor. Parent traversal, ambiguous double encoding and null bytes are refused. Roots are trusted startup configuration. Filesystem IO is synchronous and can stall the reactor on slow storage; moving this to a bounded IO subsystem is future work.

Parser/routing interfaces are independent of readiness events. OS-specific implementations are presently Linux only; the current config/root/CLI stack also uses POSIX APIs. A complete portable event interface, kqueue, IOCP, worker distribution and protocol plug-ins are future milestones, not claimed finished abstractions.

Design references: RFC 9110 https://www.rfc-editor.org/rfc/rfc9110.html and RFC 9112 https://www.rfc-editor.org/rfc/rfc9112.html. This implementation is a strict subset, not a claim of exhaustive RFC conformance.
