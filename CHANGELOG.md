# Changelog

All notable changes to this project are documented here.

## 0.1.0 — Unreleased

- Add the public Rust facade and release package metadata.
- Stabilize named `ollama`, `observability`, `evals`, `tauri`, and `programmatic`
  facade modules with Rust 1.88 as the verified minimum supported version.
- Define the seven-crate Rust publication set, including the separately
  publishable deterministic programmatic sandbox, and defer protocol/runtime SDK
  distribution from the Cargo 0.1 release.
- Define JSONL protocol v1 and the managed child-sidecar runtime.
- Add TypeScript and asyncio Python SDKs with correlated host callbacks.
- Add deterministic real-sidecar SDK integration coverage and embedded Tauri helpers.
- Add manual release-artifact staging, checksums, manifests, and consumer documentation.

## Versioning policy

First-party Rust framework crates use a unified version during the pre-1.0
series. Until 1.0, minor releases may include breaking API changes; patch
releases are reserved for compatible fixes. The sidecar protocol has its own
major/minor version and is not coupled to Cargo, npm, or PyPI package versions.
