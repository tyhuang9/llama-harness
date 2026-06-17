# llama-harness

llama-harness is a lightweight local AI harness service for managing local and gateway-backed LLM usage through a stable HTTP/SSE API. It connects directly to a locally running Ollama instance and can route cloud-provider models through a local LiteLLM Proxy.

The API is the product. The admin UI is only a local configuration and inspection dashboard.

## Project Goals

- Local-first operation.
- Direct Ollama support plus LiteLLM gateway routing for cloud providers.
- Minimal resource usage.
- API-first integration surface for other local apps.
- JSON config and optional JSONL logs instead of a database.
- Soft, low-noise admin UI for local control.

## MVP Scope

Implemented MVP capabilities:

- Rust Axum server with health, model, chat, settings, runs, and tools-placeholder endpoints.
- Ollama HTTP API integration using `http://localhost:11434` by default.
- LiteLLM Proxy integration using `http://127.0.0.1:4000` by default.
- Config persistence in `config.json`.
- Global instruction settings that can be prepended to every LLM run.
- In-memory recent run history with optional append-only `runs.jsonl`.
- Minimal React + TypeScript admin UI.
- TypeScript client SDK for external apps.

Out of scope for the MVP:

- OAuth, users, teams, billing, cloud sync, and remote deployment.
- SQLite or any other database.
- Complex auth, plugin marketplace, MCP gateway, full agent framework, or complex eval framework.

## Tech Stack

- Backend: Rust, Axum, Tokio, Serde, Reqwest.
- Frontend: React, TypeScript, Vite.
- Client SDK: TypeScript.
- Persistence: `config.json`, optional `runs.jsonl`, and in-memory state.

## Repo Structure

```text
llama-harness/
  server/        Rust backend service
  admin-ui/      Minimal local control dashboard
  clients/ts/    TypeScript client SDK for external apps
  docs/          Milestone tracking
  config.json    Local service settings
  README.md
  TODO.md
```

## Run the Rust Server

From the repo root:

```bash
cd server
cargo run
```

The server listens on `127.0.0.1:8787` by default.

Optional environment variables:

```bash
LLAMA_HARNESS_ADDR=127.0.0.1:8787
LLAMA_HARNESS_CONFIG=../config.json
LLAMA_HARNESS_RUNS_LOG=../runs.jsonl
```

## Run the Admin UI

From the repo root:

```bash
cd admin-ui
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The UI defaults to `http://127.0.0.1:8787` for the API base URL and lets you change it in Settings.

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

Run a LiteLLM Proxy locally and pin the image or package version for durable setups. The default proxy URL is:

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

In the admin UI, open Settings to enable LiteLLM, set the proxy base URL, and set an optional LiteLLM master key. Open Providers to add providers. A provider has a user-facing unique name, a LiteLLM provider type, and the environment variable that contains the provider API key:

```json
{
  "provider_type": "openai",
  "display_name": "OpenAI work",
  "api_key_env_var": "OPENAI_API_KEY",
  "api_base": null,
  "enabled": true
}
```

The provider type field can use the LiteLLM provider prefix, such as `openai`, `anthropic`, `gemini`, `openrouter`, `groq`, `mistral`, `bedrock`, `vertex_ai`, or `ollama`. Provider ids are generated internally from provider names and preserved for agents/API calls. API keys are not edited in the normal UI; users configure env vars such as `OPENAI_API_KEY`. Settings responses still mask any saved raw key as `__configured__` for compatibility.

Keep the LiteLLM proxy checkout separate from this repository. A sibling folder works well:

```bash
git clone https://github.com/BerriAI/litellm.git ../litellm
```

When `managed_config_path` is relative, llama-harness resolves it relative to `config.json`, so `../litellm/llama-harness-litellm.local.yaml` keeps proxy config with the LiteLLM checkout.

Generate a LiteLLM config from configured providers. The app automatically calls this endpoint after saving providers when LiteLLM is enabled and `managed_config_path` is configured; it can also be called directly:

```bash
curl -X POST http://127.0.0.1:8787/api/litellm/config/generate \
  -H 'content-type: application/json' \
  -d '{"output_path":"litellm.config.yaml"}'
```

Generated configs use wildcard routes so every model supported by a configured provider is available without creating a route per model. Generated configs reference environment variables, not raw provider keys:

```yaml
model_list:
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

Saving providers writes llama-harness settings and generates the LiteLLM config, but it does not hot-reload an already running LiteLLM process. Refresh or restart LiteLLM after provider changes before expecting the running proxy to pick them up.

To put local Ollama behind LiteLLM for gateway behavior such as proxy-level limits, add an enabled provider with `provider_type: "ollama"` and set API Base to your Ollama endpoint if it is not `http://localhost:11434`. llama-harness emits `ollama_chat/*` routes for that provider.

Start LiteLLM from the app or API. The server writes the managed config first, then runs `litellm --config <path> --host <host> --port <port>`. Override the executable with `LLAMA_HARNESS_LITELLM_COMMAND` when needed:

```bash
curl -X POST http://127.0.0.1:8787/api/litellm/service/start
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
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048
  }'
```

The agent model field shows suggested models for each provider and still allows a free-form model string. Existing clients can continue to call `provider: "litellm"` with an explicit LiteLLM model string or route alias.

Ollama still works the same way for local models. Existing clients that omit `provider` continue to use `default_provider`, which defaults to `ollama`.

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

## TypeScript Client

```ts
import { LlamaHarnessClient } from "@llama-harness/client";

const harness = new LlamaHarnessClient({ baseUrl: "http://127.0.0.1:8787" });

const health = await harness.health();
const models = await harness.listModels();
const result = await harness.chat({
  source_app: "note",
  prompt: "Summarize these notes.",
  instructions: "Use terse bullet points.",
});
```

## Current Limitations

- Ollama must already be installed and running locally.
- LiteLLM Proxy must already be running locally for gateway models.
- No default model is selected until one is configured.
- API token storage exists in settings, but request enforcement is not implemented in the MVP.
- Instruction settings steer model behavior, but they do not implement real tool execution.
- Run history is intentionally lightweight and capped in memory.
- Streaming is an SSE bridge over provider chat chunks and should be treated as an MVP interface.
- The admin UI is a local developer dashboard, not an embeddable product surface.
