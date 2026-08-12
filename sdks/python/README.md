# llama-harness Python SDK

`llama_harness` is an asyncio-native client for the managed Rust sidecar. Importing it does not start a process. `HarnessClient.start()` resolves an explicit runtime path, `LLAMA_HARNESS_RUNTIME_PATH`, or a package-owned matching platform binary; it never downloads a binary at runtime.

Current source builds support a workspace-built runtime passed explicitly for development and tests. Release CI stages platform-tagged wheels only after copying the matching verified Rust runtime into `llama_harness/runtime`; it never releases an empty binary placeholder.
