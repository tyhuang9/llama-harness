# Changelog

All notable changes to this project are documented here.

## 0.1.0 — Unreleased

- Add the public Rust facade and release package metadata.
- Define JSONL protocol v1 and the managed child-sidecar runtime.
- Add TypeScript and asyncio Python SDKs with correlated host callbacks.
- Add deterministic real-sidecar SDK integration coverage and embedded Tauri helpers.
- Add manual release-artifact staging, checksums, manifests, and consumer documentation.

## Versioning policy

First-party Rust framework crates use a unified version during the pre-1.0
series. Until 1.0, minor releases may include breaking API changes; patch
releases are reserved for compatible fixes. The sidecar protocol has its own
major/minor version and is not coupled to Cargo, npm, or PyPI package versions.
