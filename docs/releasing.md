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

Executable GitHub Actions are pinned to reviewed 40-character commit SHAs; the
nearby version comment is informational. Pins were resolved from each official
repository with `git ls-remote https://github.com/<owner>/<action>.git
refs/tags/<major>` on 2026-08-13. The pinned commits are: checkout
`11d5960a326750d5838078e36cf38b85af677262`, setup-node
`49933ea5288caeca8642d1e84afbd3f7d6820020`, setup-python
`a26af69be951a213d495a4c3e4e4022e16d87065`, upload-artifact
`ea165f8d65b6e75b540449e92b4886f43607fa02`, download-artifact
`d3f86a106a0bac45b974a628896c90dbdf5c8093`, configure-pages
`983d7736d9b0ae728b81ab479565c72886d7745b`, upload-pages-artifact
`56afc609e74202658d3ffba0e8f6dda462b719fa`, deploy-pages
`d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e`, and rust-toolchain stable action
`4360b52568e2003a75bf9bc1d59f33a8e3fc893c`. Review and update each SHA
deliberately when upgrading the corresponding action.

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
python -m pip install build==1.5.0 wheel==0.48.0
pwsh -File scripts/validate_release.ps1 -Platform win32-x64 -Target x86_64-pc-windows-msvc -Executable llama-harness-runtime.exe
```

Inspect `release-stage/local/artifacts/release-manifest.json` and `checksums.sha256`; the helper rejects a mismatch before returning successfully.

Both continuous and manual release validation install cargo-deny 0.20.2 and
block advisory, license, and source-policy failures. The narrow unmaintained
advisory exceptions in `deny.toml` are split by their verified Tauri paths:
Linux GTK3/proc-macro dependencies and `tauri-utils -> urlpattern -> unic-*`.
The Tauri integration maintainers own them; re-review on every Tauri or
urlpattern upgrade and immediately if any advisory severity changes.

A Tauri application receives compiled Rust dependencies inside its own installer; Ollama remains a separately installed local inference service.
