# Local Ollama validation — September 5, 2026

The current implementation completed correct, audited tool round trips with
Gemma in all 60 matched trials. LFM failed all 60 under the same prompt and
generation profile. All Adaptive trials fell back to Direct. No unauthorized,
duplicate, or unintended writes occurred, and no advanced strategy was enabled.

## Measured matrix

Frozen runner source: `7511043972cf81373c23306bc3aa29be43368e81`; suite version
`1`, fixture IDs/data retained per sample, prompt `local-task-agent-live-prompt-3`.
Every measured invocation records a clean source tree. The binary was built
with `cargo build --locked -p local-task-agent --bin live-task-agent-eval` in a
dedicated target directory and then frozen; its SHA-256 is in the environment
sidecar. Later delivery changes add documentation and evidence only.

| Installed model | Requested | Actually executed | Strict passes | Exact final state | Tool proposals / dispatches | Granted / denied approvals |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 | Direct | Direct, 30/30 | **30/30** | 30/30 | 39 / 36 | 6 / 3 |
| Gemma 4 | Adaptive | Direct, 30/30; `UnsupportedCapability` fallback | **30/30** | 30/30 | 39 / 36 | 6 / 3 |
| LFM 2.5 Thinking | Direct | Direct, 30/30 | **0/30** | 24/30 | 0 / 0 | 0 / 0 |
| LFM 2.5 Thinking | Adaptive | Direct, 30/30; `UnsupportedCapability` fallback | **0/30** | 24/30 | 0 / 0 | 0 / 0 |

Each cell contains all ten cases with three repetitions. A strict pass requires
the expected terminal behavior, exact tool arguments/counts/order, approval
decisions, exact final state, truthful final facts, output format and complete
audit binding. Missing expected mutations explain LFM's six state failures per
strategy; its unchanged stores are not successful task execution.

| Case | Gemma Direct | Gemma Adaptive → Direct | LFM Direct | LFM Adaptive → Direct |
| --- | --- | --- | --- | --- |
| No tool | 3/3 | 3/3 | 0/3 | 0/3 |
| Approved creation | 3/3 | 3/3 | 0/3 | 0/3 |
| Duplicate prevention | 3/3 | 3/3 | 0/3 | 0/3 |
| Dependent lookup/update | 3/3 | 3/3 | 0/3 | 0/3 |
| Independent reads | 3/3 | 3/3 | 0/3 | 0/3 |
| Ambiguity | 3/3 | 3/3 | 0/3 | 0/3 |
| Denied approval | 3/3 | 3/3 | 0/3 | 0/3 |
| Transient read/retry | 3/3 | 3/3 | 0/3 | 0/3 |
| Bounded read failure | 3/3 | 3/3 | 0/3 | 0/3 |
| Model-budget stop | 3/3 | 3/3 | 0/3 | 0/3 |

Gemma's six denied proposals caused zero tool dispatches. Its six dependent
updates used the ID returned by the earlier read, with one approved mutation
each. Its independent reads returned two calls in one model response, but
execution remained Direct; this is not evidence of DAG execution or speed.
All six transient-retry samples recovered, all persistent failures stayed
bounded and truthful, and all six one-model-call samples stopped at the limit.
LFM never proposed a tool, so its samples do not test approval or tool recovery.

## Accounting and limits

Settings were temperature `0`, top-p `1`, and 2,048 output tokens per model
call, with a 120-second model-call and 300-second run deadline. Case limits
further restrict the shared four-model-call/three-tool-call ceiling; the
model-budget case permits one model call. Runtime limits remain positive for
the no-tool case while its independent oracle rejects every tool proposal.
Neither seed nor thinking mode is exposed by the current provider. Top-k and
other unset options retain installed-model defaults.

| Model / requested strategy | Model calls | Input / output tokens | Median run latency |
| --- | --- | --- | --- |
| Gemma / Direct | 63 | 56,874 / 9,740 | 2,408 ms |
| Gemma / Adaptive | 63 | 56,874 / 9,746 | 2,432.5 ms |
| LFM / Direct | 39 | 36,252 / 65,526 | Not ranked: failed correctness |
| LFM / Adaptive | 39 | 36,252 / 65,526 | Not ranked: failed correctness |

All 30 Gemma pairs pass; their median Adaptive-minus-Direct difference is
`+5 ms`. Both executed the same Direct strategy, so this is not an advanced
strategy benefit. Timings are observations, not a production benchmark:
strategy order was Direct Gemma, Direct LFM, Adaptive Gemma, Adaptive LFM;
loading/caches and host activity were not controlled or balanced, and brief
deterministic verification overlapped part of the first group. Three samples
per case do not support P95 or reliability claims. All failure timings and
per-sample usage remain in the artifacts, without entering a speed ranking.

The [evidence bundle](results/2026-09-05/README.md) includes all 120 measured
samples, every failed discovery cohort, exact requests/public responses,
fixtures, state, policy/approval/dispatch audits, actual strategy events,
environment identities, a CSV, a JSON summary and checksums. Hidden reasoning
text and executable binaries are excluded.

## Source and environment

The harness under test starts at merged main
`8e52758de961d0dc289353f803a125c32f25fd18`. Fresh Git and GitHub evidence
confirmed PRs #28–#35 merged and the final implementation head
`4eeab50c1958bec7d2759a01bdeb72a478510955` is an ancestor of main. Historical
notes describing those PRs as open are superseded.

Ollama `0.33.3` was initially running at `http://127.0.0.1:11434`. During
discovery, a WSL relay began serving a separate Ollama `0.30.7` installation
at that IPv4 address. The original Windows service remained reachable at
`http://[::1]:11434`. All measured trials used that IPv6 endpoint; version
and exact model digest were rechecked before each case and after the cohort.
No services were reconfigured, no models were downloaded, and no cloud
inference was used. The machine has an AMD Ryzen 7
7800X3D, approximately 32 GiB system memory, and an NVIDIA RTX 4080 SUPER.
After calibration, Ollama reported both models resident on GPU with 4,096-token
contexts. Residency and loading affect these exploratory timings.

| Installed model | Digest | Reported relevant capabilities |
| --- | --- | --- |
| `lfm2.5-thinking:1.2b-q4_K_M` | `95bd9d45385f33bfe96d8b3651c8569e152f21f5bdb7c19894ffde650e9cf140` | Completion, tools, thinking |
| `gemma4:e4b-it-q4_K_M` | `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb` | Completion, tools, thinking |

The current harness Ollama provider declares tools, streaming and multiple
tool calls per response. It does not declare structured plans or Programmatic
AST conformance. Forced DAG and Programmatic are unsupported; Adaptive can
fall back to Direct. The model's native tool support does not establish the
missing provider contracts. Four explicit core admission checks (two models
times two unsupported strategies) exited nonzero with zero model completions
and zero tool dispatches. DAG lacked structured-plan capacity; Programmatic
first rejected the example's absent optional sandbox runtime. The Ollama
provider also lacks strict Programmatic AST conformance, so enabling that
feature alone would not establish eligibility. Speculation remains Disabled
and Programmatic promotion remains empty.

## Initial calibration, preserved separately

Before the new evaluator, a temporary example called the existing
`build_runtime` and `run_with_strategy(..., Direct)` on an empty in-memory task
store. It requested one task titled `Calibration task`, then a JSON object
with the returned ID, title and status. Settings were temperature `0`, top-p
`1`, and at most 2,048 output tokens per call. Approval was granted by the
application fixture; each model call was bounded to 120 seconds and the run
to 180 seconds. Neither thinking mode nor a seed was overridden.

| Model | Requested / executed | Model calls | Proposals / approvals | Final state | Final answer | Strict result |
| --- | --- | --- | --- | --- | --- | --- |
| LFM 2.5 Thinking | Direct / Direct | 1 | 0 / 0 | Empty, unchanged | Invented a created task without calling a tool | Fail: false success and task failure |
| Gemma 4 | Direct / Direct | 2 | 1 / 1 granted | Exactly one `task-1`, title `Calibration task`, status `open` | Correct task values inside a Markdown JSON fence | Fail: output format only |

LFM took 46,712 ms and Gemma took 27,238 ms. These are single exploratory
samples including uncontrolled loading effects, not comparative speed
evidence. The probe did not capture provider token usage. Gemma establishes
one real model → tool proposal → validation/policy/approval → store mutation
→ tool result → model answer round trip. It does not establish a general
correctness rate. LFM's failure occurred before any tool proposal, so missing
tool-result correlation cannot explain that sample.

The August 5 Promptfoo report is historical only: it used a different model
tag and unasserted cases. Its seven results graded “No assertions” are not a
correctness pass rate and are not part of this evaluation.

## Evaluator discovery corrections

The first live-evaluator revision, `e882b33391c8c36f9f15cfe48b54d829c7dae438`,
was exercised on LFM with two discovery cases before a measured cohort.
The approved update again produced zero tool calls and no state change while
claiming completion (one model call, 363 input and 971 output tokens,
44,530 ms). This repeats the false-success behavior under a second prompt.

The no-tool case failed before model contact because the new evaluator
incorrectly copied an expected maximum of zero tool calls into a runtime
limit that must be positive. Review also found that its new `get_task` read
needed explicit read-only policy handling, and that merely checking a final
`status` plus the presence of `details` could accept incorrect task facts.
These are evaluator defects, not evidence of production harness failure.
Their original failed artifacts are retained; this initial evaluator revision
is not used to claim full-suite correctness.

Two early Gemma v1 diagnostic cases also traversed a real dependent
lookup/update (one granted approval) and a denied update (zero dispatches,
unchanged store). Their weaker original oracle makes them discovery evidence,
not measured acceptance passes.

The reviewed prompt-v2 evaluator then ran all 30 Gemma Direct samples at
`5de3896818854fa7822cb83714d8a618f0edb82a`: 5/30 strict passes, but 30/30
correct state, dispatch and approval contracts. Its output instructions did
not define the exact status/outcome vocabulary required by its oracle, and
`details.outcome` invited a literal dotted field. Prompt v3 defines the full
conditional output protocol with nested JSON examples and placeholders,
without changing the strict oracle or exposing expected task facts. A
deterministic initial-request test checks the public protocol and confirms
opaque IDs/titles appear only after tool results. Three separate v3 Gemma
calibration cases passed before freezing the measured cohort.

## Defects and diagnosis

**Ollama adapter defect, fixed:** valid IPv6 loopback URLs were rejected.
`Url::host_str()` retains IPv6 brackets, which `IpAddr` did not accept. The
focused fix strips the parsed host's matching brackets before checking
`is_loopback()`. Its regression first failed on the old code specifically at
`http://[::1]:11434`, then passed with controls for IPv4, localhost, expanded
IPv6 loopback, and rejection of unspecified, remote, link-local, private,
mapped and unsupported-scheme addresses. All 17 provider tests passed.
Real IPv6 model trials establish that the corrected URL reaches the service.

**Evaluator defects, fixed:** the no-tool runtime limit, read policy,
insufficient final-fact oracle and unspecified output convention described
above were corrected before measured results. Review also strengthened
context/argument/nonce-bound approval and dispatch auditing: rejected binding
attempts are retained as hard failures, and context-free execution is blocked.
Deterministic negative tests reject false success, incorrect facts and stale
or unaudited bindings. These fixes belong to the new evaluator, not a claim
that the existing production approval broker bypassed policy.

**LFM profile failures, retained:** each strategy produced 39 completions,
including 21 responses at exactly 2,048 output tokens with empty public
content. Other responses invented unread task titles/statuses or returned
incorrect output shapes. A diagnostic captured the actual first creation
request sent by the harness and replayed that identical payload natively to
Ollama. Both upstream responses had zero tool calls, empty content, 915 input
tokens, 2,048 output tokens and `done_reason: length`. This reproduces that
failure before harness response parsing; it does not implicate tool dispatch
or tool-result correlation.

Two separate harness cases with an 8,192-token ceiling still made zero tool
calls and failed. The dependent case invented ID `TASK123` and claimed
completion while the actual `opaque-7` remained open. A separate minimal,
one-tool native prompt produced a valid `create_task` proposal at both token
ceilings (310 output tokens). Those native requests did not execute a tool
or complete a harness task and are not acceptance passes. Together the probes
support a prompt/catalog/model-profile problem, not a claim that LFM cannot
call tools at all. Prompt, catalog size, sampling and model-server behavior
were not isolated further; a new qualified profile requires its own cohort.

Upstream documents LFM Pythonic tool calls and Ollama 0.33.3 implements the
corresponding renderer/parser; a stored `{{ .Prompt }}` template alone does
not prove a broken installation because built-in rendering can bypass it.
The model card's sampling recommendations also differ from this fixed
temperature-zero profile. See the [official model card](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking#tool-use),
[pinned Ollama renderer routing](https://github.com/ollama/ollama/blob/v0.33.3/server/prompt.go#L126)
and [LFM parser](https://github.com/ollama/ollama/blob/v0.33.3/model/parsers/lfm2.go#L334).
No special tool-enable flag, capability override, model template change or
thinking-mode change was introduced.

## Reproduction

Use a clean checkout containing the frozen runner source. Select the actual
local endpoint and confirm `/api/version` and `/api/tags` match the identities
above. Create an environment sidecar from those observations; the retained
sidecar is evidence of this machine, not a substitute for fresh discovery.
Run from the repository root, with an output directory that already exists:

```powershell
$env:CARGO_TARGET_DIR = Join-Path (Get-Location) 'target/live-validation'
cargo build --locked -p local-task-agent --bin live-task-agent-eval
$runner = Join-Path $env:CARGO_TARGET_DIR 'debug/live-task-agent-eval.exe'
$common = @('--ollama-url', 'http://[::1]:11434', '--repeat', '3', '--temperature', '0', '--top-p', '1', '--output-tokens', '2048', '--max-model-call-duration-ms', '120000', '--max-run-duration-ms', '300000', '--environment-json', 'environment.json')
& $runner @common --model 'gemma4:e4b-it-q4_K_M' --strategy direct --output gemma-direct.json
& $runner @common --model 'lfm2.5-thinking:1.2b-q4_K_M' --strategy direct --output lfm-direct.json
& $runner @common --model 'gemma4:e4b-it-q4_K_M' --strategy adaptive --output gemma-adaptive.json
& $runner @common --model 'lfm2.5-thinking:1.2b-q4_K_M' --strategy adaptive --output lfm-adaptive.json
```

The recorded orchestration invoked each case separately with `--case <id>`
to retain partial progress, in YAML order. The four commands above use the
same fixtures, case order and settings. Preserve each nonzero exit and report;
LFM is expected to fail under this measured profile. Use repeated `--case`
flags to select multiple cases. For the higher-budget diagnostic, use Direct,
`--repeat 1 --case approved-mutation --case dependent-lookup-update` and
replace `--output-tokens 2048` with `8192`; keep its output separate.

Missing-service (`http://127.0.0.1:1`), missing-model and unknown-case
invocations all returned exit code 1. Explicit `--strategy declarative-plan
programmatic --case no-tool --repeat 1` also returned 1 and retained the four
unsupported results; these are intentionally unsuccessful commands.

## Deterministic baseline

On clean current main, `cargo run --locked -p xtask -- release-check` exited
successfully: formatting, Clippy, Rust tests, documentation, archive validation
and extracted consumer builds passed. Its test summaries report 491 passed
and two ignored tests. The separate local-task-agent all-targets test command
also passed. An opt-in smoke test returning early without its environment
flag is not counted as a live model trial.

After the IPv6 regression was added, the validation branch's canonical gate
passed with 492 tests and two ignored tests. The final evaluator has 20 passing
deterministic tests, including all ten cases under Direct and Adaptive plus
negative fact/audit assertions. Documentation has 12 passing tests. Commands:

```text
cargo run --locked -p xtask -- release-check
cargo test --locked -p local-task-agent --all-targets
cargo clippy --locked -p local-task-agent --all-targets -- -D warnings
cargo fmt --all -- --check
npm --prefix docs run build
npm --prefix docs test
npm --prefix docs run check
```

The SDKs were unchanged; their current-main CI results were checked, rather
than claiming a new local SDK test run. Live execution is separately invoked
and is not part of the canonical deterministic gate.

The current-main continuous-verification and documentation workflows were
initially queued, then completed successfully on this exact main commit.
All 13 [continuous-verification jobs](https://github.com/tyhuang9/llama-harness/actions/runs/33979694930)
passed, as did [documentation deployment](https://github.com/tyhuang9/llama-harness/actions/runs/33979694926).
Superseded or cancelled branch runs are not failures on current main.

## Follow-up

The [production-like benchmark protocol](strategy-benchmark-plan.md) defines
the next matched workloads and admission gates. Unsupported strategy rows
cannot supply performance or promotion evidence. Package publication, tags,
release creation, merges and advanced-strategy activation require their
separate explicit approval.

The [0.2 release-readiness assessment](release-readiness-0.2.md) distinguishes
verified package and Gemma integration behavior from the LFM profile failure,
unsupported advanced workloads, production assumptions and pending PR checks.
