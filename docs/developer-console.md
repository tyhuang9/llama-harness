# Developer console

Llama Harness Developer Console is an optional local Tauri application for inspecting one selected project workspace. It is not the runtime, a daemon, or a required background service.

## Start it

```bash
npm --prefix apps/harness-console install
npm run dev
```

The console stores only local workspace preferences. An application can embed and run `AgentRunner` without installing or opening the console.

## Connect a workspace

Project setup requires existing absolute paths for:

- the project root;
- a project-local SQLite trace database; and
- optionally, evaluation-report JSON and a YAML/JSON agent manifest inside the project root.

It also accepts a loopback Ollama URL, normally `http://127.0.0.1:11434`. Non-loopback URLs are rejected by the shared provider before model discovery.

The native bridge canonicalizes configured paths, opens the trace database in read-only mode, and validates a configured agent manifest through `llama-harness-core`. It never starts or calls the retired HTTP service.

## Inspect data

- **Models** reads direct loopback Ollama health and installed-model capabilities.
- **Agents** displays validated project-owned manifest definitions: version, prompt-version metadata, default model, allowlist, limits, optional schema, and developer-visible system instructions. An unconfigured manifest produces an empty state rather than sample agents.
- **Runs** lists redacted SQLite runs and shows the ordered causal event timeline. Raw trace payloads are discarded before data reaches the webview, even if an older database contains them.
- **Evaluations** reads saved normalized Harness reports and exposes the fixed generated Promptfoo config and raw result under the selected project's `.llama-harness` directory. It also previews or launches fixed, project-relative Harness CLI evaluation/replay commands; it cannot run arbitrary shell commands.

The standalone CLI intentionally reports when an evaluation or replay needs the embedding application’s tools, fixtures, policy, and approval adapter. The console shows that diagnostic rather than reporting a false success.

## Boundaries and privacy

The console has no generic model pull/delete controls, universal chat screen, visual agent editor, remote provider support, or generic command runner. It does not capture hidden chain-of-thought. Its local preferences do not retroactively alter trace redaction, raw capture, or retention in an existing database.
