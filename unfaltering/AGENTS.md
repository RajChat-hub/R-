# UNFALTERING engineering rules

Preserve repository files outside this project and its dedicated CI workflow. No force pushes or history rewrites. Core networking, HTTP parsing and routing remain original C++20. Do not introduce a prebuilt event loop or web server. Use audited TLS libraries when that milestone is implemented. First-party licence is pending; do not add a licence grant without owner approval.

Build with CMake and strict warnings. Run CTest after affected changes, including live socket tests. Do not replace failing tests with mocked success. Update feature status, supported configuration, security limits and validation evidence together. Never claim production readiness or cross-platform support from a successful build alone.

Build artifacts belong in ignored build directories. No credentials or private keys in Git. Keep total workspace below the directive's 16 GiB target. Do not create empty modules for unimplemented features. Do not use subagents unless the owner expressly asks.
