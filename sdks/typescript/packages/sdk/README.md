# @llama-harness/sdk

This SDK is a thin Node client for the canonical Rust `AgentRunner`. It starts an application-owned `llama-harness-runtime` child process only when `HarnessClient.start()` is called; importing this package has no process side effects.

The runtime is resolved from an explicit `runtimePath`, `LLAMA_HARNESS_RUNTIME_PATH`, or a matching `@llama-harness/runtime-<platform>-<arch>` package. The SDK never downloads a binary at runtime or searches arbitrary `PATH` entries.
