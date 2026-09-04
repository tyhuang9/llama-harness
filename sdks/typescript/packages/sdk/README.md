# @llama-harness/sdk

This SDK is a thin Node client for the canonical Rust `AgentRunner`. It starts an application-owned `llama-harness-runtime` child process only when `HarnessClient.start()` is called; importing this package has no process side effects.

The runtime is resolved from an explicit `runtimePath`, `LLAMA_HARNESS_RUNTIME_PATH`, or a matching `@llama-harness/runtime-<platform>-<arch>` package. The SDK never downloads a binary at runtime or searches arbitrary `PATH` entries.

The optional `strategy` field (`adaptive`, `direct`, `declarative_plan`, or
`programmatic`) requires negotiated protocol 1.1. Omit it when connected to a
1.0 runtime. The managed sidecar does not configure a programmatic sandbox and
therefore rejects `programmatic`; that mode is available only to explicitly
configured embedded Rust hosts.

Set `agent.outputSchema` to request bounded JSON Schema validation. The Rust
runtime rejects invalid schemas and external references before model execution.
