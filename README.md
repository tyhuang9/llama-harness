# llama-harness

llama-harness is a local AI harness for routing chat requests through a stable Rust HTTP/SSE API. It talks directly to a locally running Ollama instance and can manage a local LiteLLM Proxy sidecar for OpenAI-compatible cloud or gateway providers.

The API is the product. The admin UI is only a local configuration and inspection dashboard.

## What It Offers

- Stable local API endpoints for health, model inventory, app capabilities, runs, streaming runs, settings, providers, and audit.
- Direct local Ollama support with `http://localhost:11434` as the default endpoint.
- App-managed LiteLLM runtime for gateway-backed models without vendoring LiteLLM into this repo.
- Provider-scoped agent defaults so Agents can use local Ollama or a saved LiteLLM provider.
- Local admin UI for settings, providers, apps, agents, tools, permissions, runs, audit, and model testing.
- Tauri desktop shell for running the local dashboard as a desktop app.
- TypeScript client SDK for other local apps.
- JSON config, app-data `.env` secrets, and JSONL logs instead of a database.

## Current Boundaries

- This is local-first infrastructure, not a hosted multi-user SaaS.
- Ollama must already be installed and running for direct local models.
- LiteLLM is downloaded into ignored/generated Python environments; LiteLLM source, virtualenvs, `site-packages`, and generated runtime files are not committed.
- No OAuth, users, teams, billing, cloud sync, database, plugin marketplace, MCP gateway, or full eval framework is included.
- Agent/task screens are local operator workflows; they do not yet run a full autonomous task runner.

## Tech Stack

- Backend: Rust, Axum, Tokio, Serde, Reqwest.
- Frontend: React, TypeScript, Vite.
- Desktop: Tauri.
- Local model provider: Ollama.
- Gateway provider: LiteLLM Proxy installed from `requirements-litellm.txt`.
- Client SDK: TypeScript.
- Persistence: `config.json`, `config/*.json`, OS app-data `.env` and LiteLLM YAML files, optional JSONL run/audit logs, and in-memory recent state.

## Repo Structure

```text
llama-harness/
  server/        Rust backend service
  admin-ui/      Local admin dashboard
  clients/ts/    TypeScript client SDK for external apps
  desktop/       Tauri desktop shell
  docs/          Milestone tracking
  scripts/       Dev and deployment helpers
  bundled/       Ignored generated deployment runtime output
  config/        Local model, agent, app, and tool catalog JSON
  config.json    Local service settings
  requirements-litellm.txt
  README.md
  TODO.md
```

The repository tracks a safe baseline `config.json` so new checkouts have LiteLLM settings and a local Ollama-through-LiteLLM provider. Treat your working `config.json` as local runtime state after that. To hide future local edits from normal Git status:

```bash
git update-index --skip-worktree config.json
```

To intentionally change the committed baseline later:

```bash
git update-index --no-skip-worktree config.json
```

## Run the App

The normal development flow starts the Tauri desktop shell, the Vite admin UI,
and the Axum backend from the repo root:

```bash
npm run dev
```

Use the focused commands below when you only need one part of the stack.

### Rust Server

From the repo root:

```bash
npm run api:dev
```

The server listens on `127.0.0.1:8787` by default.

Optional environment variables:

```bash
LLAMA_HARNESS_ADDR=127.0.0.1:8787
LLAMA_HARNESS_CONFIG=config.json
LLAMA_HARNESS_CONFIG_DIR=config
LLAMA_HARNESS_RUNS_LOG=runs.jsonl
LLAMA_HARNESS_AUDIT_LOG=logs/audit.jsonl
```

### Admin UI

From the repo root:

```bash
npm install
npm run web:dev
```

Open the Vite URL shown in the terminal. The UI defaults to `http://127.0.0.1:8787` for the API base URL and lets you change it in Settings.

## LiteLLM Runtime Setup

LiteLLM is managed as a local sidecar process by the Rust backend. The repository
does not vendor LiteLLM source, Python virtualenvs, `site-packages`, or generated
runtime files. Developers create a local ignored venv, and release builds create
an ignored bundled runtime that Tauri includes in the installer. The pinned
runtime dependency is `litellm[proxy]==1.89.1`.

Install Python 3.10 through 3.13 first. For development on Linux or macOS:

```bash
cd /home/tyhuang/Projects/llama-harness
./scripts/setup-litellm-dev.sh
npm run dev
```

For development on Windows PowerShell:

```powershell
cd path\to\llama-harness
.\scripts\setup-litellm-dev.ps1
npm run dev
```

The setup script creates `.venv-litellm/`, installs `requirements-litellm.txt`,
prints the resolved Python path, and verifies the installed LiteLLM version.

For deployment builds on Linux or macOS:

```bash
cd /home/tyhuang/Projects/llama-harness
./scripts/build-litellm-runtime.sh
npm run tauri:build
```

For deployment builds on Windows PowerShell:

```powershell
cd path\to\llama-harness
.\scripts\build-litellm-runtime.ps1
npm run tauri:build
```

The runtime build script recreates `bundled/litellm-runtime/`, installs
LiteLLM from `requirements-litellm.txt`, verifies the import, and leaves the
runtime for Tauri to bundle. Both `.venv-litellm/` and
`bundled/litellm-runtime/` are ignored.

## Configure Ollama

The default Ollama endpoint is stored in `config.json`:

```json
{
  "ollama_endpoint": "http://localhost:11434",
  "default_model": null,
  "instructions": {
    "enabled": false,
    "system_prompt": "",
    "tool_context": ""
  }
}
```

You can edit `config.json` directly or update settings through:

```bash
curl -X PUT http://127.0.0.1:8787/api/settings \
  -H 'content-type: application/json' \
  -d '{"ollama_endpoint":"http://localhost:11434","default_model":"llama3.2"}'
```

## Configure LiteLLM

LiteLLM mode keeps llama-harness focused on one internal model-provider abstraction while letting LiteLLM handle OpenAI, Anthropic, OpenRouter, Gemini, and future cloud providers. Ollama remains the local direct provider. Cloud services should usually be added as LiteLLM provider records, not one route per model.

The Rust backend starts LiteLLM as an internal sidecar and talks to it over localhost. The default proxy URL is:

```text
http://127.0.0.1:4000
```

Typical environment variables:

```bash
LITELLM_MASTER_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

API keys can be provided through the process environment or the app-data `.env`
file managed by llama-harness. The app-data directory is controlled by
`LLAMA_HARNESS_DATA_DIR` when set; otherwise it uses the OS app-data location
for `llama-harness`. Generated LiteLLM YAML files and saved `.env` secrets live
there, outside the repository.

In the admin UI, open Settings to enable LiteLLM and set the proxy base URL. Open Providers to add providers. A provider has a user-facing unique name, a LiteLLM provider type, and the environment variable that contains the provider API key:

```json
{
  "provider_type": "openai",
  "display_name": "OpenAI work",
  "api_key_env_var": "OPENAI_API_KEY",
  "api_base": null,
  "enabled": true
}
```

The provider type field can use the LiteLLM provider prefix, such as `openai`, `anthropic`, `gemini`, `openrouter`, `groq`, `mistral`, `bedrock`, `vertex_ai`, or `ollama`. Provider ids are generated internally from provider names and preserved for agents/API calls. Settings responses mask saved raw keys as `__configured__`.

Generate a LiteLLM config from configured providers. The app automatically calls this endpoint after saving providers when LiteLLM is enabled and `managed_config_path` is configured; it can also be called directly:

```bash
curl -X POST http://127.0.0.1:8787/api/litellm/config/generate \
  -H 'content-type: application/json' \
  -d '{"output_path":"litellm.config.yaml"}'
```

Relative config paths are resolved inside the app-data directory, not the repo.
Generated configs use wildcard routes so every model supported by a configured
provider is available without creating a route per model. They also include
default example aliases and reference environment variables, not raw provider
keys:

```yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: os.environ/OPENAI_API_KEY

  - model_name: local-llama
    litellm_params:
      model: ollama/llama3.1
      api_base: http://localhost:11434

  - model_name: openai/*
    litellm_params:
      model: openai/*
      api_key: os.environ/OPENAI_API_KEY

  - model_name: anthropic/*
    litellm_params:
      model: anthropic/*
      api_key: os.environ/ANTHROPIC_API_KEY

  - model_name: gemini/*
    litellm_params:
      model: gemini/*
      api_key: os.environ/GEMINI_API_KEY

  - model_name: ollama_chat/*
    litellm_params:
      model: ollama_chat/*
      api_base: http://localhost:11434

litellm_settings:
  check_provider_endpoint: true

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Saving providers writes llama-harness settings, writes app-data secrets, generates the LiteLLM config, and restarts the managed LiteLLM process when llama-harness owns it. If another process already owns the LiteLLM port, llama-harness will not stop it.

To put local Ollama behind LiteLLM for gateway behavior such as proxy-level limits, add an enabled provider with `provider_type: "ollama"` and set API Base to your Ollama endpoint if it is not `http://localhost:11434`. llama-harness emits `ollama_chat/*` routes for that provider.

Start LiteLLM from the app or API. The server writes the managed config first,
resolves Python from `LLAMA_HARNESS_LITELLM_PYTHON`,
`LLAMA_HARNESS_LITELLM_RUNTIME_DIR`, or `.venv-litellm/`, then runs the
runtime's `litellm` console entrypoint with
`--config <path> --host <host> --port <port>`. Startup readiness is checked at
`/health/readiness` with safe fallbacks and uses the configured LiteLLM
`timeout_ms` window:

```bash
curl -X POST http://127.0.0.1:8787/api/litellm/service/start
```

External apps should call the Rust backend, not LiteLLM directly. The stable
chat endpoint is `/api/chat`; OpenAI-compatible callers can use the forwarding
endpoint:

```bash
curl -X POST http://127.0.0.1:8787/api/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hello."}]}'
```

Explicit LiteLLM routes remain supported in config and API responses for legacy or advanced cases such as aliases, fallbacks, load balancing, rate limits, or custom model rewrites. They are intentionally not part of the normal app UI because provider-level wildcard routing covers the common case.

Test a LiteLLM provider with a model name:

```bash
curl -X POST http://127.0.0.1:8787/api/providers/litellm/test \
  -H 'content-type: application/json' \
  -d '{"provider_id":"openai_main","model":"gpt-4o","message":"Say hello from llama-harness."}'
```

Call chat through a configured provider. The UI shows provider names, while the agent or caller stores the internal provider id and a plain model name; llama-harness converts that to the LiteLLM model string, such as `openai/gpt-4o` or `anthropic/claude-sonnet-4-0`:

```bash
curl -X POST http://127.0.0.1:8787/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "provider": "openai_main",
    "model": "gpt-4o",
    "messages": [
      { "role": "user", "content": "Write a quick project summary." }
    ],
    "generation": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 2048
    }
  }'
```

The agent model field shows provider-scoped model choices. Ollama choices come
from the local inventory; saved LiteLLM providers use provider-specific
suggestions. Existing API clients can continue to call `provider: "litellm"`
with an explicit LiteLLM model string or route alias.

Ollama still works the same way for local models. Existing clients that omit `provider` continue to use `default_provider`, which defaults to `ollama`.

## App-Agent Policy Model

The harness owns model selection, agent routing, tool permissions, and run policy. External apps should identify themselves with an app id and let llama-harness resolve what they are allowed to use.

Top-level concepts are separate:

- Models: local Ollama model records in `config/models.json`.
- Agents: reusable behavior profiles in `config/agents.json`.
- Apps: external client applications and their allowed agents/tools in `config/apps.json`.
- Tools: visible app capabilities in `config/tools.json`; apps execute their own domain tools and return results to the run.
- Runs: model execution records in `runs.jsonl`.
- Audit: policy decisions and denied requests in `logs/audit.jsonl`.

Agents are not nested under apps. An agent can be reused by multiple apps. An app chooses which agents it may use and which one is the default.

The seeded Note policy is:

```json
{
  "id": "note",
  "name": "Note",
  "defaultAgentId": "note-assistant",
  "allowedAgentIds": ["note-assistant"],
  "allowedToolIds": [
    "note.getCurrentPage",
    "note.getSelectedBlocks",
    "note.searchPages",
    "note.createBlock",
    "note.updateBlock",
    "note.deleteBlock",
    "note.moveBlock",
    "note.createPage",
    "note.renamePage",
    "note.openPage"
  ],
  "enabled": true
}
```

Note should connect with:

```json
{
  "appId": "note"
}
```

Then discover its resolved assignment:

```bash
curl http://127.0.0.1:8787/apps/note/capabilities
```

Example response:

```json
{
  "appId": "note",
  "appName": "Note",
  "defaultAgent": {
    "id": "note-assistant",
    "name": "Note Assistant",
    "description": "Helps summarize, analyze, and reason about notes."
  },
  "allowedAgents": [
    {
      "id": "note-assistant",
      "name": "Note Assistant",
      "description": "Helps summarize, analyze, and reason about notes."
    }
  ],
  "tools": [
    {
      "id": "note.getCurrentPage",
      "name": "Get Current Page",
      "description": "Read the active Note page and, optionally, its visible text blocks.",
      "riskLevel": "low",
      "enabled": true
    }
  ],
  "model": {
    "id": "ollama-default",
    "name": "llama3.2",
    "provider": "ollama",
    "modelName": "llama3.2",
    "status": "available"
  }
}
```

If no `agentId` is supplied when creating a run, llama-harness uses the app's default agent:

```bash
curl -X POST http://127.0.0.1:8787/runs \
  -H 'content-type: application/json' \
  -d '{
    "appId": "note",
    "agentId": null,
    "input": "Summarize the current page.",
    "context": {
      "pageId": "page_123",
      "pageTitle": "Meeting Notes",
      "selectedText": "",
      "blocks": []
    }
  }'
```

The response includes the resolved app, agent, model, output, duration, and `toolRequests` when the model asks the app to run tools. Apps submit those results to `POST /runs/:runId/tool-results`; the response either completes the run or returns another `requires_action` round. `POST /runs/stream` provides the same app policy flow over SSE. `/api/apps`, `/api/apps/:appId/capabilities`, `/api/runs`, `/api/runs/:runId/tool-results`, `/api/runs/stream`, and `/api/audit` are equivalent API-prefixed routes for existing clients.

## Configure Global Instructions

Global instructions are saved in `config.json` and prepended as a system message for every chat, model test, and streaming chat request when enabled.

```bash
curl -X PUT http://127.0.0.1:8787/api/settings \
  -H 'content-type: application/json' \
  -d '{
    "instructions": {
      "enabled": true,
      "system_prompt": "You are a careful local assistant.",
      "tool_context": "summarize_note: summarize note text\nextract_actions: return action items"
    }
  }'
```

Apps can also pass request-specific instructions without changing the global config:

```json
{
  "source_app": "note",
  "prompt": "Extract action items from this note.",
  "instructions": "Return only a checklist."
}
```

## API Examples

Health:

```bash
curl http://127.0.0.1:8787/health
```

List local Ollama models:

```bash
curl http://127.0.0.1:8787/api/models
```

Set the default model:

```bash
curl -X POST http://127.0.0.1:8787/api/models/default \
  -H 'content-type: application/json' \
  -d '{"model":"llama3.2"}'
```

Send a chat request:

```bash
curl -X POST http://127.0.0.1:8787/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "source_app": "note",
    "prompt": "Extract action items from this note.",
    "instructions": "Return only actionable checklist items.",
    "model": "llama3.2"
  }'
```

Stream a chat request over SSE:

```bash
curl -N -X POST http://127.0.0.1:8787/api/chat/stream \
  -H 'content-type: application/json' \
  -d '{"prompt":"Write a short local status summary.","model":"llama3.2"}'
```

Read recent runs:

```bash
curl http://127.0.0.1:8787/api/runs
```

Read recent audit decisions:

```bash
curl http://127.0.0.1:8787/api/audit
```

## TypeScript Client

```ts
import { LlamaHarnessClient } from "@llama-harness/client";

const harness = new LlamaHarnessClient({ baseUrl: "http://127.0.0.1:8787" });

const health = await harness.health();
const models = await harness.listModels();
const capabilities = await harness.appCapabilities("note");
const result = await harness.run({
  appId: "note",
  input: "Summarize these notes.",
  context: { pageId: "page_123" },
});
```

## Current Limitations

- Ollama must already be installed and running locally.
- LiteLLM setup requires either the dev venv from `scripts/setup-litellm-dev.*` or the bundled runtime from `scripts/build-litellm-runtime.*`.
- No default model is selected until one is configured.
- API token storage exists in settings, but request enforcement is not implemented yet.
- Instruction settings and tool capability records steer model behavior, but they do not implement real tool execution.
- Run history is intentionally lightweight and capped in memory.
- Audit is intentionally lightweight and omits full prompts, secrets, and full app context.
- Streaming is an SSE bridge over provider chat chunks.
- The admin UI is a local developer dashboard, not an embeddable product surface.
