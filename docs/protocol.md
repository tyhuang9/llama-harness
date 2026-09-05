# Sidecar protocol

Protocol v1 is newline-delimited JSON over the private stdin/stdout pipes of an SDK-owned `llama-harness-runtime` child process. It is not an HTTP API, TCP listener, or global daemon.

Package `0.2.0` uses protocol 1.1 while retaining the specified 1.0 fallback.
The package version is not a wire-version substitute: the first exchange records
the SDK name and version in `client_hello` and the runtime package version in
`runtime_hello`. Release validation requires those identities and the Cargo,
npm, and Python package metadata to match the requested release exactly.

Standard output is exclusively protocol frames. Runtime diagnostics use standard error. Every envelope has a protocol version, request correlation ID, optional run ID, type, and typed payload. Events are monotonic per run.

The Rust protocol crate owns the canonical wire contracts and explicit bounds. The checked-in envelope schema and golden handshake fixtures are under [`protocol/`](../protocol); detailed compatibility rules are in [`protocol/compatibility/v1.md`](../protocol/compatibility/v1.md).

The initial runtime advertises `supports_output_deltas: false`: the current canonical model-provider contract is non-streaming. It does not synthesize token deltas from a completed response.

Guarded speculation is a same-process Direct-runner optimization and does not
extend protocol v1. No speculation configuration, candidate, mode, readiness,
counter, argument, result, or raw error is projected into commands, callbacks,
events, or terminal results. A protocol-backed provider without a useful
finalized index-0 stream boundary gains no speculative overlap.

The runtime accepts only `client_hello` as its first command. It generates run and callback IDs, accepts each callback response at most once, and fails a run closed on callback validation failure, timeout, cancellation, or disconnect. Every run-scoped runtime message has a monotonically increasing `run_sequence`; the terminal message is last.

An SDK must stop when the first response is not a matching `runtime_hello`, when
the protocol major is incompatible, or when a later envelope drifts from the
negotiated minor. It must not fall back to HTTP, an arbitrary `PATH` executable,
or a downloaded runtime. These are compatibility and provenance failures, not
recoverable transport retries.
