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

LiteLLM mode keeps llama-harness focused on one internal model-provider abstraction while letting LiteLLM handle OpenAI, Anthropic, OpenRouter, Gemini, and future cloud providers. Ollama remains the local direct provider; cloud services should usually be added as LiteLLM routes rather than as separate Rust clients.

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

In the admin UI, open Settings to enable LiteLLM, set the proxy base URL, set an optional LiteLLM master key, and choose a default LiteLLM model alias. Open Models to add model routes. Route examples:

```text
openai:gpt-4o        -> openai/gpt-4o
anthropic:claude     -> anthropic/<model>
openrouter:claude    -> openrouter/<provider>/<model>
gemini:flash         -> gemini/<model>
```

Keep the LiteLLM proxy checkout separate from this repository. A sibling folder works well:

```bash
git clone https://github.com/BerriAI/litellm.git ../litellm
```

When `managed_config_path` is relative, llama-harness resolves it relative to `config.json`, so `../litellm/llama-harness-litellm.local.yaml` keeps proxy config with the LiteLLM checkout.

Generate a LiteLLM config from configured routes:

```bash
curl -X POST http://127.0.0.1:8787/api/litellm/config/generate \
  -H 'content-type: application/json' \
  -d '{"output_path":"litellm.config.yaml"}'
```

Generated configs reference environment variables, not raw provider keys:

```yaml
model_list:
  - model_name: openai:gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
```

Test a LiteLLM model route:

```bash
curl -X POST http://127.0.0.1:8787/api/providers/litellm/test \
  -H 'content-type: application/json' \
  -d '{"model":"openai:gpt-4o","message":"Say hello from llama-harness."}'
```

Call chat through LiteLLM:

```bash
curl -X POST http://127.0.0.1:8787/api/chat \
  -H 'content-type: application/json' \
  -d '{
    "provider": "litellm",
    "model": "openai:gpt-4o",
    "messages": [
      { "role": "user", "content": "Write a quick project summary." }
    ],
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048
  }'
```

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
