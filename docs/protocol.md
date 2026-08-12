# Sidecar protocol

Protocol v1 is newline-delimited JSON over the private stdin/stdout pipes of an SDK-owned `llama-harness-runtime` child process. It is not an HTTP API, TCP listener, or global daemon.

Standard output is exclusively protocol frames. Runtime diagnostics use standard error. Every envelope has a protocol version, request correlation ID, optional run ID, type, and typed payload. Events are monotonic per run.

The Rust protocol crate owns the canonical wire contracts and explicit bounds. The checked-in envelope schema and golden handshake fixtures are under [`protocol/`](../protocol); detailed compatibility rules are in [`protocol/compatibility/v1.md`](../protocol/compatibility/v1.md).

The initial runtime advertises `supports_output_deltas: false`: the current canonical model-provider contract is non-streaming. It does not synthesize token deltas from a completed response.

The runtime accepts only `client_hello` as its first command. It generates run and callback IDs, accepts each callback response at most once, and fails a run closed on callback validation failure, timeout, cancellation, or disconnect. Every run-scoped runtime message has a monotonically increasing `run_sequence`; the terminal message is last.
