# llama-harness

llama-harness is a lightweight local AI harness service for managing local LLM usage through a stable HTTP/SSE API. It connects to a locally running Ollama instance and gives other local apps a small control plane for model selection, request execution, run visibility, and settings.

The API is the product. The admin UI is only a local configuration and inspection dashboard.

## Project Goals

- Local-first operation.
- Ollama-only model provider for the MVP.
- Minimal resource usage.
- API-first integration surface for other local apps.
- JSON config and optional JSONL logs instead of a database.
- Simple black-on-white admin UI for local control.

## MVP Scope

Implemented MVP capabilities:

- Rust Axum server with health, model, chat, settings, runs, and tools-placeholder endpoints.
- Ollama HTTP API integration using `http://localhost:11434` by default.
- Config persistence in `config.json`.
- In-memory recent run history with optional append-only `runs.jsonl`.
- Minimal React + TypeScript admin UI.
- TypeScript client SDK for external apps.

Out of scope for the MVP:

- Cloud model providers such as OpenAI, Anthropic, Gemini, or hosted inference APIs.
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
  "default_model": null
}
```

You can edit `config.json` directly or update settings through:

```bash
curl -X PUT http://127.0.0.1:8787/api/settings \
  -H 'content-type: application/json' \
  -d '{"ollama_endpoint":"http://localhost:11434","default_model":"llama3.2"}'
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
});
```

## Current Limitations

- Ollama must already be installed and running locally.
- No default model is selected until one is configured.
- API token storage exists in settings, but request enforcement is not implemented in the MVP.
- Run history is intentionally lightweight and capped in memory.
- Streaming is a direct SSE bridge over Ollama chat chunks and should be treated as an MVP interface.
- The admin UI is a local developer dashboard, not an embeddable product surface.

