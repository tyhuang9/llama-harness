# Promptfoo integration

Promptfoo is an optional developer dependency, pinned in `tools/promptfoo`. It does not run in the embedded runtime, is not packaged with an application, and never becomes an agent tool.

The bundled wrapper is a concrete adapter for the checked-in `local-task-agent` example. For every Promptfoo case it starts a fresh `local-task-agent promptfoo-adapter` process. That adapter loads the validated suite case and fixture, constructs the example's registered task tools, policy, approval handler, local Ollama provider, and redacted SQLite event sink, then runs the full `AgentRunner`. It is not a text-only model wrapper.

## Install and generate

From the repository root, install the pinned development dependency:

```bash
npm --prefix tools/promptfoo install
npm --prefix tools/promptfoo exec promptfoo -- --version
```

Generate and inspect a config without contacting Ollama:

```bash
cargo run -p llama-harness-cli -- eval promptfoo evals/local-task-agent/suite.yaml --model ollama:your-installed-model --show-config
```

`--model` must be an installed `ollama:<model>` ID. The wrapper rejects non-Ollama models, a model name without a suffix, a suite for a different agent, a non-loopback Ollama URL, and an output directory that leaves the current project root. A different application must supply its own concrete adapter rather than rely on this example's task tools, fixtures, policy, or approvals.

## Run and inspect results

Run the generated config explicitly:

```bash
cargo run -p llama-harness-cli -- eval promptfoo evals/local-task-agent/suite.yaml --model ollama:your-installed-model --run
```

The command invokes only the pinned `npm`/Promptfoo executable with fixed arguments; it does not use a shell or accept arbitrary program names. Each provider call is limited to a declared suite case and the configured local model. It records the following project-local artifacts:

- `.llama-harness/generated/promptfooconfig.yaml` — inspectable generated Promptfoo configuration;
- `.llama-harness/generated/agent-provider.mjs` — the custom provider source;
- `.llama-harness/results/promptfoo-results.json` — raw Promptfoo output;
- `.llama-harness/results/promptfoo-traces.sqlite` — redacted Harness trace events; and
- `.llama-harness/results/promptfoo-normalized-report.json` — Harness-normalized assertions with run and trace IDs.

Promptfoo's own pass/fail display reports provider execution. The normalized Harness report evaluates the suite's deterministic tool, policy, approval, state, and limit assertions; use it as the correctness result. Inspect a normalized report with:

```bash
cargo run -p llama-harness-cli -- eval results .llama-harness/results/promptfoo-normalized-report.json
```

If normalization finds failed Harness assertions, the Promptfoo CLI command exits nonzero after writing the report. This prevents a provider-response success from being treated as an agent-evaluation success.

The Developer Console exposes the generated config and raw result from their fixed `.llama-harness` locations, as well as the normalized report when its results path is configured. These artifacts omit raw model request/response persistence and hidden reasoning.
