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
