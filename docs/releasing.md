# Releasing llama-harness

No package or release artifact is published by this repository's normal CI checks. Releases are explicit, manually approved operations.

## Rust crates.io runbook

This section is the manual runbook for the seven Rust crates. It is independent
of the deferred sidecar/SDK artifact workflow below. Do not combine their
authorizations or assume that a successful SDK validation publishes a crate.

The initial Rust release uses the unified `0.1.0` version:
`llama-harness-programmatic-sandbox`, `llama-harness-core`,
`llama-harness-ollama`, `llama-harness-observability`, `llama-harness-tauri`,
`llama-harness-evals`, and the `llama-harness` facade.
Protocol/runtime, SDKs, Promptfoo integration, scripted runtime, examples,
developer console, CLI, and release helper are not published in this milestone.

### Prerequisites and release gate

Before publishing, record the exact reviewed commit on clean `main` as
`SOURCE_COMMIT`. The worktree must have no unreviewed or uncommitted changes,
and the requested version must match the workspace/package metadata and the
changelog or prepared release notes. For `0.1.0`, recheck all seven crate names
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

```powershell
$SOURCE_COMMIT = (git rev-parse HEAD).Trim()
pwsh -File scripts/release/check-rust-release.ps1 -Version 0.1.0 -SourceCommit $SOURCE_COMMIT
cargo run --locked -p xtask -- release-check
```

The preflight validates the exact seven-crate set, unified version, Rust 1.88
MSRV, nonempty changelog entry, clean worktree, and source commit. The helper
runs formatting, linting, tests, and Rust documentation, creates all seven
`.crate` archives, rejects unexpected, unsafe, or oversized archive contents,
and checks extracted packages with the supported facade feature configurations.

Manually dispatch `.github/workflows/release-rust.yml` from `main` with version
`0.1.0` and the exact 40-character `SOURCE_COMMIT`. The workflow rejects a
different current `main` commit. Confirm every portable platform,
publishable-crate platform, package, policy, documentation, API, and fresh Rust
1.88 job passes against that commit. This workflow has read-only repository
permissions, receives no registry credentials, uploads no release artifacts,
and cannot publish, tag, create a release, or change owners. Sidecar and SDK
validation remains in the separate `.github/workflows/release.yml` workflow and
cannot block this Rust release gate.

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

### Programmatic sandbox package audit

`llama-harness-programmatic-sandbox` is a separately published `no_std` plus
`alloc` library, not an operating-system process sandbox. Its public boundary
accepts only the versioned AST; verified bytecode remains private. The crate
has `#![forbid(unsafe_code)]`, no `build.rs`, no build dependencies, and normal
dependencies limited to `serde` and `serde_json` with their default features
disabled. The release helper rejects unexpected archive paths and compiles the
extracted standalone consumer. Recheck the package boundary before publishing:

```powershell
cargo check --locked --package llama-harness-programmatic-sandbox --no-default-features
cargo tree --locked --package llama-harness-programmatic-sandbox --no-default-features --edges normal
cargo package --locked --list --package llama-harness-programmatic-sandbox
```

Do not describe this crate as providing filesystem, network, process, or tool
authority. The same-process host remains the only broker and policy authority.

### Publication and registry checks

Publish manually in the following exact dependency order. A layer is complete only when
the exact `0.1.0` package is resolvable from the crates.io index, its package
page is available, and a fresh external consumer has built against it. Do not
use workspace paths, Git dependencies, a `[patch.crates-io]` table, cached lock
files, or `cargo update` to make a consumer pass.

Immediately before every dry-run and publish command below, rerun the preflight
and confirm it still reports `SOURCE_COMMIT`:

```powershell
pwsh -File scripts/release/check-rust-release.ps1 -Version 0.1.0 -SourceCommit $SOURCE_COMMIT
```

#### 1. Programmatic sandbox

Dry-run and publish `llama-harness-programmatic-sandbox`:

```powershell
cargo publish --locked --dry-run --package llama-harness-programmatic-sandbox
cargo publish --locked --package llama-harness-programmatic-sandbox
```

#### 2. Core

Wait for `llama-harness-programmatic-sandbox = 0.1.0` to appear in the index,
build a fresh exact-version consumer, then dry-run and publish
`llama-harness-core`:

```powershell
cargo publish --locked --dry-run --package llama-harness-core
cargo publish --locked --package llama-harness-core
```

#### 3. Direct core integrations

Wait for `llama-harness-core = 0.1.0` to appear in the index, create a new
temporary consumer with `llama-harness-core = "=0.1.0"`, and build it. Then
dry-run and publish each direct integration:

```powershell
cargo publish --locked --dry-run --package llama-harness-ollama
cargo publish --locked --dry-run --package llama-harness-observability
cargo publish --locked --dry-run --package llama-harness-tauri
cargo publish --locked --package llama-harness-ollama
cargo publish --locked --package llama-harness-observability
cargo publish --locked --package llama-harness-tauri
```

#### 4. Evaluations

Wait for all three integration versions to be index-visible, build fresh
exact-version consumers for each newly available crate, then dry-run and
publish `llama-harness-evals`:

```powershell
cargo publish --locked --dry-run --package llama-harness-evals
cargo publish --locked --package llama-harness-evals
```

#### 5. Facade

Wait for `llama-harness-evals = 0.1.0` to be index-visible, build a fresh
`llama-harness-evals = "=0.1.0"` consumer, then dry-run and publish the
`llama-harness` facade:

```powershell
cargo publish --locked --dry-run --package llama-harness
cargo publish --locked --package llama-harness
```

Wait for `llama-harness = 0.1.0` to be index-visible, then run the final facade
consumer verification below.

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
`observability`, `evals`, `tauri`, and `programmatic`), and `--all-features`.

Verify the published `0.1.0` documentation pages for all seven crates:

1. `https://docs.rs/llama-harness-programmatic-sandbox/0.1.0`
2. `https://docs.rs/llama-harness-core/0.1.0`
3. `https://docs.rs/llama-harness-ollama/0.1.0`
4. `https://docs.rs/llama-harness-observability/0.1.0`
5. `https://docs.rs/llama-harness-tauri/0.1.0`
6. `https://docs.rs/llama-harness-evals/0.1.0`
7. `https://docs.rs/llama-harness/0.1.0`

Add the approved backup crates.io login to all seven crates, then verify each
owner list. Repeat these two commands for every crate name:

```powershell
cargo owner --add <backup-login> llama-harness-core
cargo owner --list llama-harness-core
```

Only after all seven pages render, all seven owner lists are confirmed, and the
final facade consumer passes, create the annotated `v0.1.0` tag pointing
exactly to `SOURCE_COMMIT`, push that tag, and create the GitHub release from
the checked-in notes:

```powershell
git tag --annotate v0.1.0 $SOURCE_COMMIT --message "llama-harness v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --verify-tag --title "llama-harness v0.1.0" --notes-file .github/releases/v0.1.0.md
```

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
a GitHub release until the entire seven-crate set has passed the final checks.

The rollback boundary is the registry publish: source changes, tags, and a
GitHub release can be reverted or corrected, but a published crate cannot be
removed or overwritten. A GitHub release or tag never makes an incomplete or
bad registry release safe; communicate the issue and ship the smallest safe
forward fix.

For this Programmatic release, rollback is also behavior-sensitive: retain the
execution-ID SQLite migration and trace schema together with core. Reverting
only the trace reader can merge reused public run IDs or lose stable legacy
identity selection. Before a forward rollback release, verify Direct and
Adaptive compatibility, Programmatic's value-free sandbox-error mapping, and
the bounded completion-only Programmatic path; streaming remains intentionally
deferred until a separately bounded contract is released.

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
