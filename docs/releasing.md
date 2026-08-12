# Releasing llama-harness

No package or release artifact is published by this repository's normal CI checks. Releases are explicit, manually approved operations.

## Before a first release

Create a crates.io account and token, verify intended package-name availability, and store the token only in the protected CI secret store. Published crates.io versions are immutable: correct a bad release with a new version or, when appropriate, yank it; yanking does not delete the published source package.

The initial publishable Rust crates use a unified `0.1.0` version. The development-only Promptfoo integration, examples, developer console, and release helper are intentionally not published. The CLI is also local-only while it depends on development-only Promptfoo support.

## Local release verification

Run the release helper from a clean checkout:

```bash
cargo run -p xtask -- release-check
```

It runs formatting, linting, tests, docs, and `cargo package --list` for each publishable crate. `cargo package --list` exposes exactly what a crates.io source package would contain without publishing it.

The helper deliberately does not claim that `cargo publish --dry-run` can completely validate the very first release: crates with unpublished first-party dependencies cannot be resolved by crates.io until their dependencies exist. After the dependency crates have been staged or published in order, run `cargo publish --dry-run --package <crate>` for each crate before the actual publication command.

## Publication order

The order follows workspace dependencies and is verified by the release checklist:

1. `llama-harness-core`
2. `llama-harness-protocol`
3. `llama-harness-evals`, `llama-harness-observability`, and `llama-harness-ollama`
4. `llama-harness`

Later protocol, Tauri, runtime, npm, PyPI, and binary-release steps are documented alongside the sidecar distribution work. A Tauri application receives compiled Rust dependencies inside its own installer; Ollama remains a separately installed local inference service.
