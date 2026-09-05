# Production-like strategy benchmark plan

This is a follow-up protocol, not a benchmark result or an authorization to
enable a strategy. Begin only after the functional live suite meets its
correctness and effect checks for the specific provider/model combination.

## Admission and invariants

- Use an immutable source commit, installed model digest, Ollama version,
  application prompt and fixture versions, generation settings, and explicit
  requested strategy. Record actual selected strategy and every fallback.
- Rebuild synthetic state for every sample. Never use user task stores,
  application credentials, or remote tools. Audit actual tool execution and
  approval decisions independently of model proposals.
- Unauthorized, duplicate, and unintended effects are hard failures. Report
  attempted unauthorized actions separately from effects successfully blocked
  by the broker. Reject false final success claims even when the store is safe.
- Keep provider declarations unchanged. An unsupported strategy is an
  admission result, not a successful workload run. Adaptive-to-Direct samples
  belong to the Direct execution cohort.
- Keep Programmatic promotion empty and speculation Disabled. A capability
  declaration alone does not supply workload competence or promotion evidence.

## Matched workloads

| Workload | Synthetic fixture and independent oracle | Comparison prerequisite |
| --- | --- | --- |
| Dependent task update | Resolve an opaque task ID by title, update once after approval, verify exact store | Direct and Adaptive tool round trips pass |
| Independent reads | Read 2, 8, and 32 opaque task records, preserve identity and exact values | Declarative-plan provider support before claiming DAG speed |
| Fan-out and reduction | Read independent partitions; return exact count and status histogram | Supported strategy, safe read-only declarations and bounded concurrency |
| Bounded loop/filter | Filter 10, 100, and 1,000 task records by explicit status and aggregate exact counts | Strict AST conformance, configured sandbox, validated workload evidence |
| Large intermediate results | Three 256 KiB synthetic partitions with unique canaries; return only a compact aggregate | Verify reduction, transcript limits and absence of intermediate canaries in final synthesis |
| Recovery | One read fails before success; a permanent failure remains within the same fixed attempt budget | Record failed attempts, retry/recovery phase, final truthfulness and unchanged writes |
| Denial and limits | Deny a write; exhaust model/tool budgets before additional effects | Zero unauthorized dispatch and no replay after uncertain effects |

Use actual task data and result sizes, not fixed sleeps, to describe workload
cost. Optional controlled tool latency must be separately labeled as injected
latency and excluded from claims about actual application I/O.

## Measurement protocol

First retain a small discovery cohort to debug fixture, prompt and adapter
issues. Freeze their versions before the measured cohort; preserve failed
discovery data and report it separately. Run one model at a time and record
other inference activity, model residency, context configuration and machine
resources. Keep cold-load and warm-run latency separate.

Run matched cases with identical fixture and generation settings. Balance
strategy order within each model using a recorded schedule. Start with 30
repetitions per supported cell as a screening cohort; use additional samples
and an explicit uncertainty analysis before making tail-latency or production
claims. This sample count is a proposed protocol, not an existing product gate.

Report all samples and failure categories first. For matched pairs that both
pass task, final-state, final-answer and safety assertions, compare elapsed
time, model calls, input/output tokens, tool attempts/dispatches, and recovery
counts. Report median paired differences and sample counts; use sufficient
data and confidence intervals for P95 conclusions. Failed or unsupported rows
must never improve a latency average by being silently dropped.

## Bounded executable screening

The checked-in screening cohort has only two cases: the unchanged
`dependent-lookup-update` control and the opt-in `independent-reads-8` case.
The latter requires eight opaque `get_task` reads, unchanged state, no approval
or mutation, and an exact eight-record JSON envelope. It does not alter the
ten-case `live-suite.yaml` or its prompt version.

Build the evaluator first, then run the standard-library driver from a clean,
frozen worktree with a new output directory:

```text
python evals/local-task-agent/benchmark_driver.py --binary <worktree>/target/debug/live-task-agent-eval --suite <worktree>/evals/local-task-agent/benchmark-suite.yaml --environment-json <explicit-environment.json> --source-root <worktree> --output-dir <new-external-output-directory> --model gemma4:e4b-it-q4_K_M --ollama-url http://[::1]:11434 --cohort-wall-seconds 2400
```

Each child is an argument-list invocation with `shell=False` and exactly one
case, strategy, and repeat:

```text
<binary> --model gemma4:e4b-it-q4_K_M --ollama-url http://[::1]:11434 --suite <worktree>/evals/local-task-agent/benchmark-suite.yaml --case <case> --repeat 1 --strategy <direct|adaptive> --temperature 0 --top-p 1 --output-tokens 2048 --max-model-calls 9 --max-tool-calls 8 --max-model-call-duration-ms 120000 --max-run-duration-ms 300000 --environment-json <explicit-environment.json> --output <new-external-output-directory>/<phase>/<sample>/report.json
```

The driver records the executable SHA-256, source commit/clean state, suite and
environment digests before every child. It rejects a reused output directory,
identity drift, dirty source, missing artifact, bad artifact identity, timeout,
or failed strict evaluator/effect audit. It retains each raw JSON report and
stdout/stderr, prints flushed `phase`/`case`/`strategy` completion lines, and
stops admission before measured runs if any discovery or warmup row fails.

The fixed schedule has four discovery rows, four separate warmup rows, then 30
pairs per workload (120 measured samples): odd pairs run Direct then Adaptive;
even pairs reverse both strategy and case order. A configurable 40-minute
cohort deadline bounds each child by the remaining time and records every
unstarted row as missing before exiting nonzero.

The reducer reports failures before all planned, completed, and missing rows,
then separate discovery, warmup, and measured summaries. Headline metrics use
only measured rows. Paired medians use evaluator `duration_ms`, never outer
subprocess elapsed time, and include a pair only when both requested-strategy
rows fully pass the strict oracle and effect/audit checks. The report includes
paired N, excluded pairs, requested and actual strategies/fallbacks, model
calls, tokens, attempts, dispatches, audit violations, and mutations. It makes
no P95 or production-performance claim.

The deterministic driver tests require no model or service:

```text
python evals/local-task-agent/run_benchmark_driver_tests.py
```

## Promotion and rollback

Programmatic requires provider strict AST V1 support and sandbox configuration
before a real workload can be measured. Promote only the demonstrated workload
class after separate approval, with no correctness or safety regression and a
measured benefit. Remove that class from the host allowlist to disable future
Adaptive promotion; do not replay a run that may already have produced effects.

Speculation has separate existing gates in
[the speculation guide](../../docs/speculative-tool-calling.md): at least
1,000 exact per-tool Shadow observations, explicit activation, and matched
Active/Disabled evidence. Never speculate writes. The functional live suite
does not satisfy those gates.

The release-readiness assessment must distinguish deterministic library and
packaging checks, model-specific functional competence, unsupported strategy
coverage, production assumptions, and pending remote CI. No benchmark outcome
authorizes a merge, package publication, tag, or release.
