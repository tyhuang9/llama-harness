# 0.2 release-readiness assessment — September 5, 2026

Release rehearsals passed on main `f545bd524c622689fc92daf51e6c0e7f17c63175`
after the user-approved merge of [PR #38](https://github.com/tyhuang9/llama-harness/pull/38).
The initial manual rehearsal exposed missing Rust formatter/linter components;
that one-line workflow fix passed 34 CI jobs before merge. No registry upload,
tag, or GitHub release was performed. The user's manual crates.io setup and
first publication remain pending.

| Surface | Verified evidence | Remaining condition |
| --- | --- | --- |
| Main integration | PR #37 live validation and PR #38 workflow fix merged; final-main [continuous verification](https://github.com/tyhuang9/llama-harness/actions/runs/33990319828) passed 13/13 jobs | Recheck the exact approved release source before publication |
| Rust 0.2 contracts and packaging | Final-main [Rust release rehearsal](https://github.com/tyhuang9/llama-harness/actions/runs/33990339151) passed 8/8 jobs: three-platform crate checks, Rust 1.88 resolution, API/docs, policy and consumers | Initial publication, dependency indexing, exact-version registry consumers, owners and docs.rs |
| Runtime and SDK artifacts | Final-main [manual rehearsal](https://github.com/tyhuang9/llama-harness/actions/runs/33990343206) passed 7/7 jobs; ten downloaded payloads passed manifest, content and checksum inspection | Authorized npm/PyPI identity and publication, if those channels are intended |
| Installed Windows SDKs | Fresh offline Node and Python installations used their packaged runtime, verified matching bytes, and completed protocol 1.1 as version 0.2.0 without an override | Registry installation and production embedding remain separate checks |
| Live evaluation infrastructure | 23 deterministic example tests and seven driver regressions; canonical gate passed 492 tests, two ignored, docs, archives and extracted consumers on frozen code `51d0a60` | Review the benchmark PR and its final-head CI; keep inference opt-in |
| Gemma functional suite | 30/30 Direct and 30/30 requested Adaptive strict passes across ten guided cases, including denial and recovery | Broader tasks and deployment-specific failure injection |
| Gemma workload screening | 120/120 measured passes across an approved dependent update and eight independent reads; four discovery and four warmup samples separate | Two synthetic workloads do not establish general reliability or tail latency |
| Actual strategy | Every live and benchmark Adaptive sample executed Direct | No DAG, Programmatic, speculative execution, or strategy speedup claim |
| LFM under the earlier profile | 0/30 Direct and 0/30 Adaptive; safe unchanged stores and zero proposals | Qualify a new profile before deployment; not admitted to the benchmark |
| DAG and Programmatic | Four earlier explicit admission checks rejected unsupported configurations before model/tool calls | Conforming provider, sandbox where required, workload evidence and promotion approval |
| Speculation | Disabled throughout; Programmatic promotion list empty | Existing per-tool Shadow and activation gates plus matched evidence and approval |

The [functional results](live-ollama-2026-09-05.md) and
[bounded benchmark results](benchmark-ollama-2026-09-05.md) use installed Gemma,
Ollama 0.33.3, explicit tool/output instructions, a four-tool catalog and fresh
in-memory fixtures. All 120 measured benchmark samples passed an independent
raw-evidence audit: 720 model calls, 600 tool dispatches, and 60 intended writes
with correctly bound approvals. Unauthorized, duplicate and unintended effects
were zero. The eight-read workload used nine serial model calls per sample;
its timings establish a Direct baseline only. Thirty matched pairs per case
do not establish P95 bounds or production reliability.

Benchmark code was frozen at `51d0a60a1f29231e5762d8b2f830e74c6a972962`,
which descends from release-rehearsal source `f545bd5`. Source, binary, suite
and environment hashes were checked before every invocation; final source,
binary, Ollama version and model digest matched the pre-cohort snapshot.
The additions are opt-in evaluation tooling, deterministic tests and docs.
Release artifacts were built from `f545bd5`, not a later evidence commit.
Select and revalidate the final approved source before immutable publication.

## Publication work still required

Follow the [release runbook](../../docs/releasing.md), including ownership and
backup-owner requirements. Complete the user-owned crates.io step, publish in
dependency order and wait for indexing between layers. Core and upper-layer
registry dry runs and fresh exact-version consumers depend on those first
publications. Verify packages, owners and docs.rs before tagging. npm/PyPI
access and distribution must be confirmed separately if included. A name's
404 response does not prove namespace ownership.

Eight clean-main Rust archives and the runtime/SDK artifact set are prepared
locally with checksums. The sandbox's final-source
`cargo publish --locked --dry-run` passed and explicitly aborted upload.
Cargo retained a warning for an unselected yanked `chacha20 0.10.1` lock entry;
the all-feature/all-target inverse tree found no selected path and the existing
policy gate passed without an exemption. Recheck registry/advisory state before
upload.

On Windows, the downloaded wheel was installed into a new Python 3.12.10 venv
with `pip install --no-index --no-deps`; downloaded SDK/runtime tarballs were
installed into fresh Node 24.15.0/npm 12.0.1 dependencies with
`npm install --offline --ignore-scripts --omit=dev --no-audit --no-fund`.
Both SDKs resolved their installed runtime, matched Windows runtime SHA-256
`5d07a3c49c9ec676df567fef00ced1d721f0f329d5c7d2e73c7fab9e29fcbc03`,
started, negotiated protocol 1.1 and closed. These checks made no model calls
and do not substitute for registry-consumer or deployed application checks.

Rollback of the evaluation work is to stop invoking the opt-in driver. No
service configuration, user store, library strategy gate or release version
was changed. Immutable package recovery follows the runbook's new-version
policy. These results do not authorize merging a new PR, publication, a tag,
or advanced-strategy activation.
