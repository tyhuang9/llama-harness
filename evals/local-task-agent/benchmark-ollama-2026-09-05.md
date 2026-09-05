# Bounded local workload benchmark — September 5, 2026

Installed Gemma passed **120/120 measured samples**, with zero unauthorized,
duplicate or unintended effects. All actual execution was Direct, including
requested Adaptive. This screening establishes a baseline for two guided
synthetic workloads; it provides no evidence of a DAG or Programmatic speedup.

Four discovery and four warmup samples also passed. They are retained
separately and excluded from measured totals and paired timings. An independent
audit recomputed every raw sample's contracts and the aggregate results.

## Measured results

Each cell contains 30 samples. Strategy order was balanced within matched pairs;
even pairs also reversed case order. No measured samples or pairs were excluded.

| Workload | Requested | Actual | Strict passes | Median duration | Model calls | Input / output tokens | Tool dispatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dependent lookup and approved update | Direct | Direct | 30/30 | 3,997 ms | 90 | 82,920 / 15,735 | 60 |
| Dependent lookup and approved update | Adaptive | Direct | 30/30 | 4,008 ms | 90 | 82,920 / 15,735 | 60 |
| Eight independent reads | Direct | Direct | 30/30 | 11,487 ms | 270 | 200,580 / 45,810 | 240 |
| Eight independent reads | Adaptive | Direct | 30/30 | 11,488.5 ms | 270 | 200,580 / 45,810 | 240 |

Calls, tokens and dispatches above are cell totals. Across measured samples:
720 model calls, 567,000 input tokens, 123,090 output tokens, 600 tool attempts
and dispatches, 60 correctly approved intended updates and zero audit violations.
The update case used three model calls per sample. The eight-read case used
nine: one request/result round trip for each record, then a final answer.
All 60 Adaptive rows recorded `UnsupportedCapability` fallback to Direct.

For the 30 fully passing pairs per case, the median **paired**
Adaptive-minus-Direct duration was **+43.5 ms** for the update and **−19.5 ms**
for eight reads. These are medians of within-pair differences, not differences
between cell medians. Median paired call, token and dispatch differences were
zero. These small differences between runs of the same execution strategy do
not establish a performance benefit. No P95 or confidence-bound claim is made.

The metric is the evaluator's run duration, excluding outer process startup
and CLI prerequisite checks. The complete discovery/warmup/measured driver
finished in 1,061.016 seconds within its 2,400-second cap. The model started
cold in discovery and was resident before measured runs; discovery load time
is not included in the headline timings.

## Profile and audit

- Source: `51d0a60a1f29231e5762d8b2f830e74c6a972962`, clean before every child.
- Evaluator SHA-256: `03ede7b272b3fe26111b66bee8a768ee11e685ecf568fe0d85bc1d84d7d80f42`.
- Model: `gemma4:e4b-it-q4_K_M`, digest `c6eb396dbd5992bbe3f5cdb947e8bbc0ee413d7c17e2beaae69f5d569cf982eb`.
- Ollama 0.33.3 at `http://[::1]:11434`; Windows, Ryzen 7 7800X3D, 32 GiB-class RAM, RTX 4080 SUPER 16 GiB, driver 610.88.
- Temperature 0, top-p 1, maximum output 2,048 tokens; seed not exposed, thinking and top-k use provider/model defaults. Warm/end residency reported context 4,096 and only Gemma loaded.
- Suite `local-task-agent-strategy-benchmark`, prompt `local-task-agent-benchmark-prompt-1`; the existing dependent-case output instructions and ten-case live suite remain unchanged.
- Maximum nine model calls, eight tool calls, 120-second model-call and 300-second run deadlines. Existing stricter case-specific contracts still apply.
- Programmatic promotion list empty; speculation Disabled; provider capability declarations unchanged.

Source, executable, suite and environment hashes were checked before each
invocation. Native Ollama version and model digest were captured before and
after the cohort and matched; these are endpoint snapshots, not per-call native
identity attestations. No local compiler processes were present at the boundary
snapshots, and this task performed no local builds or package installations
during the cohort. Other host activity was not continuously measured.

Every sample used a fresh in-memory store. The update required an actual list
result to resolve an opaque ID, one policy-approved update, a correctly bound
approval and exact final state. Eight reads required each opaque ID exactly
once in any order, exact returned facts, unchanged state, no approvals and no
mutation. Both cases required model-generated final JSON consistent with actual
results. Independent auditing checked proposals, tool results, dispatch order
or multiset as appropriate, approval bindings, state, final response provenance,
schedule completeness, hashes and recomputed totals/paired metrics.

The first full independent audit used the LF Git-blob suite hash against the
CRLF Windows input and rejected all rows for that identity expectation alone.
The retained byte comparison proves the 2,185-byte input becomes the exact
2,132-byte frozen Git blob after CRLF normalization. The corrected portable
audit verifies both hashes and passes all 128 raw rows. The initial rejection
and diagnosis are preserved; no model run, source or raw result was changed.

## Reproduction and limits

See the [executable protocol](strategy-benchmark-plan.md) for exact build/run
arguments, deadlines, admission checks and deterministic test entry point.
The [retained evidence](results/2026-09-05-benchmark/README.md) includes all
128 raw reports, stdout/stderr, invocation schedule, environment snapshots,
frozen inputs, independent audit and checksums. It supports offline re-auditing
without a running model. Fresh model reproduction requires the recorded model
and environment; reruns are new cohorts and must not be pooled silently.

The model was admitted using the earlier [ten-case functional suite](live-ollama-2026-09-05.md).
LFM failed that profile and was not admitted here. This benchmark uses explicit
tool/output guidance and fast in-memory tools. It does not measure open-ended
planning, application I/O, batching/DAG speed, loops, reduction of large results,
streaming, injected failures or production concurrency. The remaining workloads
and supported-provider prerequisites stay in the benchmark protocol.

The frozen benchmark's canonical gate passed 492 tests with two ignored, plus
docs, eight package archives and extracted consumers. Its 23 example tests and
seven model-free driver regressions also passed. Package release rehearsals
ran on the preceding main source `f545bd5`; see the
[release-readiness assessment](release-readiness-0.2.md) for exact workflows,
fresh installed SDK checks and the pending manual registry steps.
