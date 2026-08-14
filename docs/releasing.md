# Releasing llama-harness

No package or release artifact is published by this repository's normal CI checks. Releases are explicit, manually approved operations.

## Before a first release

Create a crates.io account and token, verify intended package-name availability, and store the token only in the protected CI secret store. Published crates.io versions are immutable: correct a bad release with a new version or, when appropriate, yank it; yanking does not delete the published source package.

The initial publishable Rust crates use a unified `0.1.0` version. The development-only Promptfoo integration, scripted test runtime, examples, developer console, and release helper are intentionally not published. The CLI is also local-only while it depends on development-only Promptfoo support.

## Local release verification

Run the release helper from a clean checkout:

```bash
cargo run -p xtask -- release-check
```

It runs formatting, linting, tests, docs, and `cargo package --list` for each publishable crate. `cargo package --list` exposes exactly what a crates.io source package would contain without publishing it.

Continuous verification runs the Rust checks on Windows, macOS, and Linux; blocks Rust advisory, license, and source-policy failures; and tests, builds, and inspects both SDK packages. The developer console's npm audit is reported but does not block a runtime release: its current Vite/esbuild and nanoid findings are development-tooling-only and are tracked separately in `TODO.md`.

The helper deliberately does not claim that `cargo publish --dry-run` can completely validate the very first release: crates with unpublished first-party dependencies cannot be resolved by crates.io until their dependencies exist. After the dependency crates have been staged or published in order, run `cargo publish --dry-run --package <crate>` for each crate before the actual publication command.

## Publication order

The order follows workspace dependencies and is verified by the release checklist:

1. `llama-harness-core`
2. `llama-harness-protocol`
3. `llama-harness-evals`, `llama-harness-observability`, and `llama-harness-ollama`
4. `llama-harness-runtime` and `llama-harness-tauri`
5. `llama-harness`

## Sidecar SDK distribution

The release workflow at `.github/workflows/release.yml` is manual-dispatch only. It builds `llama-harness-runtime` for supported Windows x64, macOS arm64, and Linux x64 targets, generates checksums and a machine-readable manifest, and stages a matching platform-specific npm runtime package plus Python wheel. It does not build model images, pull models, or package Ollama.

The workflow defaults to validation mode. Set its `publish` input only after a reviewer confirms the version, generated manifest, checksums, package contents, registry credentials, and release notes. Keep the publication order: Rust dependencies first, runtime artifacts next, npm/PyPI SDK packages after their corresponding artifacts, then the facade. Registry publication remains an operator action; no normal CI job publishes packages.

For a local platform rehearsal without publishing, run:

```powershell
python -m pip install build wheel
pwsh -File scripts/validate_release.ps1 -Platform win32-x64 -Target x86_64-pc-windows-msvc -Executable llama-harness-runtime.exe
```

Inspect `release-stage/local/artifacts/release-manifest.json` and `checksums.sha256`; the helper rejects a mismatch before returning successfully.

A Tauri application receives compiled Rust dependencies inside its own installer; Ollama remains a separately installed local inference service.
