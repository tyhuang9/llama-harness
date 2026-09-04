# llama-harness-ollama

`llama-harness-ollama` is the direct Ollama provider for
[`llama-harness`](https://crates.io/crates/llama-harness). It connects only to
loopback Ollama endpoints and supports streamed model responses without using
the sidecar runtime.

```toml
[dependencies]
llama-harness = { version = "0.2.0", features = ["ollama"] }
```

Ollama must already be running and the requested model must already be
installed. The provider does not pull models and rejects non-loopback URLs.

The minimum supported Rust version is 1.88. This crate is licensed under the
MIT License; see `LICENSE`.
