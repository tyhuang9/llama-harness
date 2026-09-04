# Changelog

All notable changes to this project are documented here.

## 0.2.0 — Unreleased

- Release the unified 0.2.0 first-party Rust workspace surface and matching
  managed TypeScript and Python SDK package metadata.
- Stabilize the checked protocol v1.1 SDK/runtime handshake, including exact
  package identity reporting and compatible v1.0 fallback.
- Add deterministic release gates for Cargo metadata, runtime hello identity,
  protocol acceptance, SDK package contents, and matching npm/Python artifacts.
- Make Adaptive the default quality-first dispatcher across Direct,
  declarative DAG, and explicitly evaluation-promoted Programmatic workloads,
  while preserving Direct as the shared broker and fallback boundary.
- Add provider-neutral strict structured-output requests, cohort-based
  safety/correctness readiness with P50/P95 ranking, and explicit
  non-speculative recovery, retry, and fallback paths.
- Document the 0.1-to-0.2 migration, release sequencing, and immutable-package
  rollback boundaries.

## 0.1.0 — Superseded development baseline

This unpublished compatibility baseline was superseded by the unified 0.2.0
release candidate.

- Add the public Rust facade and release package metadata.
- Stabilize named `ollama`, `observability`, `evals`, `tauri`, and `programmatic`
  facade modules with Rust 1.88 as the verified minimum supported version.
- Define the eight-crate Rust publication set, including the separately
  publishable deterministic programmatic sandbox, and defer protocol/runtime SDK
  distribution from the initial Cargo release.
- Define JSONL protocol v1 and the managed child-sidecar runtime.
- Add TypeScript and asyncio Python SDKs with correlated host callbacks.
- Add deterministic real-sidecar SDK integration coverage and embedded Tauri helpers.
- Add manual release-artifact staging, checksums, manifests, and consumer documentation.

## Versioning policy

First-party Rust framework crates use a unified version during the pre-1.0
series. Until 1.0, minor releases may include breaking API changes; patch
releases are reserved for compatible fixes. The sidecar protocol has its own
major/minor version and is not coupled to Cargo, npm, or PyPI package versions.
