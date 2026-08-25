# Releasing llama-harness

No package or release artifact is published by this repository's normal CI checks. Releases are explicit, manually approved operations.

## Rust crates.io runbook

This section is the manual runbook for the six Rust crates. It is independent
of the deferred sidecar/SDK artifact workflow below. Do not combine their
authorizations or assume that a successful SDK validation publishes a crate.

The initial Rust release uses the unified `0.1.0` version:
`llama-harness-core`, `llama-harness-ollama`, `llama-harness-observability`,
`llama-harness-tauri`, `llama-harness-evals`, and the `llama-harness` facade.
Protocol/runtime, SDKs, Promptfoo integration, scripted runtime, examples,
developer console, CLI, and release helper are not published in this milestone.

### Prerequisites and release gate

Before publishing, record the exact reviewed commit on clean `main` as
`SOURCE_COMMIT`. The worktree must have no unreviewed or uncommitted changes,
and the requested version must match the workspace/package metadata and the
changelog or prepared release notes. For `0.1.0`, recheck all six crate names
on crates.io immediately before the first publish; a name conflict or an
existing `0.1.0` means stop and choose a forward version rather than trying to
overwrite it.

Use a crates.io account with the required publish access, and add or verify a
backup owner under the project's approved ownership policy before the final
tag and GitHub release. Keep the token only in the protected CI secret store or
Cargo's credential store. Never paste it into a shell command, repository file,
issue, release note, terminal capture, or CI output; revoke and replace it if
exposure is suspected.

Run both validation gates against `SOURCE_COMMIT` before any publish action:

```bash
cargo run --locked -p xtask -- release-check
```

The helper runs formatting, linting, tests, and Rust documentation, creates all
six `.crate` archives, rejects unexpected, unsafe, or oversized archive
contents, and checks extracted packages with the supported facade feature
configurations. Also manually dispatch `.github/workflows/release.yml` with
`publish` left `false`, confirm the requested version and generated artifacts,
and review its Rust release-validation result. That workflow validates sidecar
and SDK artifacts as well; it does not publish Cargo crates.

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

The helper deliberately does not claim that `cargo publish --dry-run` can
completely validate the first layer: unpublished first-party dependencies are
not yet resolvable from crates.io. Registry-dependent dry-runs begin only after
the preceding layer is available in the crates.io index.

### Publication and registry checks

Publish manually in this exact dependency order. A layer is complete only when
the exact `0.1.0` package is resolvable from the crates.io index, its package
page is available, and a fresh external consumer has built against it. Do not
use workspace paths, Git dependencies, a `[patch.crates-io]` table, cached lock
files, or `cargo update` to make a consumer pass.

1. Publish `llama-harness-core`.
2. Wait for `llama-harness-core = 0.1.0` to appear in the index, create a new
   temporary consumer with `llama-harness-core = "=0.1.0"`, and build it.
   Then dry-run the next layer: `llama-harness-ollama`,
   `llama-harness-observability`, and `llama-harness-tauri`.
3. Publish `llama-harness-ollama`, `llama-harness-observability`, and
   `llama-harness-tauri`. Wait for all three exact versions to be index-visible,
   build fresh exact-version consumers for the newly available crates, and
   dry-run `llama-harness-evals`.
4. Publish `llama-harness-evals`. Wait for its exact version to be index-visible,
   build a fresh `llama-harness-evals = "=0.1.0"` consumer, and dry-run the
   `llama-harness` facade.
5. Publish `llama-harness`. Wait for `llama-harness = 0.1.0` to be index-visible,
   then run the final facade consumer verification below.

For every dry-run, use the exact package selected for the next layer, for
example `cargo publish --dry-run --package llama-harness-evals`. Only run the
matching `cargo publish --package <crate>` after the dry-run, source commit,
artifact contents, registry identity, and version have been reconfirmed. These
are manual commands; this repository's workflows do not perform the publication.

### Final facade and documentation verification

Create a fresh, outside-workspace consumer for every final profile. Its manifest
must contain only `llama-harness = "=0.1.0"`; it must not contain path, Git, or
patch overrides. Generate a new lockfile and build the default configuration,
`--no-default-features`, each individual supported feature (`ollama`,
`observability`, `evals`, and `tauri`), and `--all-features`.

Verify the published `0.1.0` documentation pages for all six crates:

1. `https://docs.rs/llama-harness-core/0.1.0`
2. `https://docs.rs/llama-harness-ollama/0.1.0`
3. `https://docs.rs/llama-harness-observability/0.1.0`
4. `https://docs.rs/llama-harness-tauri/0.1.0`
5. `https://docs.rs/llama-harness-evals/0.1.0`
6. `https://docs.rs/llama-harness/0.1.0`

Only after all six pages render, the backup owner is confirmed, and the final
facade consumer passes, create the annotated `v0.1.0` tag pointing exactly to
`SOURCE_COMMIT`. Create the GitHub release from that tag and use the checked-in
`.github/releases/v0.1.0.md` notes without editing them in the release form.

### Failure handling and rollback boundaries

Published crates.io versions are immutable: never retry a version by attempting
to overwrite it. Prefer a forward fix with a new version. Yank only when the
published version should be discouraged for new resolutions and consumers can
reasonably move forward; yanking does not delete the crate, repair existing
lockfiles, or undo downloads.

If a layer partially releases, stop before its dependents, record which exact
packages are visible, and diagnose from fresh consumers. Recover by publishing
the remaining safe packages only when their exact dependency graph is still
correct; otherwise publish a coordinated forward version. Do not tag or create
a GitHub release until the entire six-crate set has passed the final checks.

The rollback boundary is the registry publish: source changes, tags, and a
GitHub release can be reverted or corrected, but a published crate cannot be
removed or overwritten. A GitHub release or tag never makes an incomplete or
bad registry release safe; communicate the issue and ship the smallest safe
forward fix.

## Sidecar SDK distribution

The release workflow at `.github/workflows/release.yml` is manual-dispatch only. It builds `llama-harness-runtime` for supported Windows x64, macOS arm64, and Linux x64 targets, generates checksums and a machine-readable manifest, and stages a matching platform-specific npm runtime package plus Python wheel. It does not build model images, pull models, or package Ollama.

The workflow defaults to validation mode. Set its `publish` input only after a
reviewer confirms the version, generated manifest, checksums, package contents,
registry credentials, and release notes. The deferred SDK sequence is separate:
publish runtime artifacts first, then the matching npm/PyPI packages. It neither
precedes nor blocks the Cargo sequence above. Each sequence requires its own
explicit human authorization; no normal CI job publishes packages.

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
