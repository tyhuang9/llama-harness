# Local Ollama validation — September 5, 2026

## Source and environment

The harness under test starts at merged main
`8e52758de961d0dc289353f803a125c32f25fd18`. Fresh Git and GitHub evidence
confirmed PRs #28–#35 merged and the final implementation head
`4eeab50c1958bec7d2759a01bdeb72a478510955` is an ancestor of main. Historical
notes describing those PRs as open are superseded.

Ollama `0.33.3` was already running at `http://127.0.0.1:11434`. No models were
downloaded and no cloud inference was used. The machine has an AMD Ryzen 7
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
missing provider contracts. Speculation remains Disabled and Programmatic
promotion remains empty.

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

## Deterministic baseline

On clean current main, `cargo run --locked -p xtask -- release-check` exited
successfully: formatting, Clippy, Rust tests, documentation, archive validation
and extracted consumer builds passed. Its test summaries report 491 passed
and two ignored tests. The separate local-task-agent all-targets test command
also passed. An opt-in smoke test returning early without its environment
flag is not counted as a live model trial.

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
