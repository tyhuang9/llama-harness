# llama-harness Python SDK

`llama_harness` is an asyncio-native client for the managed Rust sidecar. Importing it does not start a process. `HarnessClient.start()` resolves an explicit runtime path, `LLAMA_HARNESS_RUNTIME_PATH`, or a package-owned matching platform binary; it never downloads a binary at runtime.

Current source builds support a workspace-built runtime passed explicitly for development and tests. Release CI stages platform-tagged wheels only after copying the matching verified Rust runtime into `llama_harness/runtime`; it never releases an empty binary placeholder.

The optional `strategy` argument (`adaptive`, `direct`, `declarative_plan`, or
`programmatic`) requires negotiated protocol 1.1. Omit it when connected to a
1.0 runtime. The managed sidecar does not configure a programmatic sandbox and
therefore rejects `programmatic`; that mode is available only to explicitly
configured embedded Rust hosts.

Agent mappings may supply `output_schema` or `outputSchema`. The SDK sends the
schema to the Rust runtime, which applies the bounded JSON Schema contract and
rejects external references before model execution.
