# SDK architecture

The TypeScript and Python SDKs are lifecycle managers for one local child
process. They do not embed a second agent framework and do not communicate with
a service. Calling `start()` launches `llama-harness-runtime` with private
stdin/stdout/stderr pipes; importing either package does nothing.

```text
Node/Python host -- JSONL stdin/stdout -- child runtime -- AgentRunner -- loopback Ollama
      |                                      |
      +-- host tool/policy/approval callbacks-+
```

Protocol v1 requires `client_hello` first. The runtime assigns run IDs and
callback IDs, serializes stdout through one bounded writer, and treats stdout
corruption or process exit as a typed SDK failure. It never parses stderr as a
protocol channel. Every tool request still goes through the Rust runner's tool
registry, JSON Schema validation, agent allowlist, policy decision, approval,
deadline, and cancellation checks before the host callback is requested.

For the 0.2.0 release, each `client_hello` reports the installed SDK's package
identity and `runtime_hello` reports the child executable's Cargo identity. The
release gate starts the built child and compares both against npm/Python and
Cargo metadata before packaging. That check establishes artifact provenance; it
does not let an SDK bypass ordinary protocol-version negotiation.

SDK runtime lookup accepts an explicit path, `LLAMA_HARNESS_RUNTIME_PATH`, or a
matching package-owned platform artifact. It never searches arbitrary `PATH`
locations or downloads code. A sidecar parent owns its child lifetime: `close`
sends shutdown, EOF/crash cancels active work, and cancellation cannot undo a
host side effect already under way.

Provider inspection is explicit and outside the run lifecycle:
`health`/`listModels` in TypeScript and `health`/`list_models` in Python issue
their own typed commands and never allocate a run, invoke a tool, or alter an
agent transcript.

See [`protocol/compatibility/v1.md`](../protocol/compatibility/v1.md) for the
wire compatibility policy and the language-specific guides for examples.
