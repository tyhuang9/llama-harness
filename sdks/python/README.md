# llama-harness Python SDK

`llama_harness` is an asyncio-native client for the managed Rust sidecar. Importing it does not start a process. `HarnessClient.start()` resolves an explicit runtime path, `LLAMA_HARNESS_RUNTIME_PATH`, or a package-owned future platform binary; it never downloads a binary at runtime.

Current source builds support a workspace-built runtime passed explicitly for development and tests. Release wheels will be platform-tagged only after signed runtime artifacts are copied into them by release CI.
