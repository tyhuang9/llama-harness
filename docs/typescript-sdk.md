# TypeScript SDK

Install `@llama-harness/sdk` with its reviewed, matching platform runtime
package, or set `LLAMA_HARNESS_RUNTIME_PATH` during local development. The SDK
requires Node 20 or later and never downloads a runtime.

```ts
import { defineTool, HarnessClient } from "@llama-harness/sdk";

const search = defineTool({
  id: "notes.search", name: "Search notes", description: "Search local notes",
  argumentsSchema: { type: "object", required: ["query"], properties: { query: { type: "string" } } },
  risk: "low", idempotent: true, readOnly: true,
  execute: async ({ query }) => ({ matches: await findLocalNotes(query) }),
});

await using client = await HarnessClient.start({ provider: { kind: "ollama" } });
const run = await client.run({
  agent: { id: "notes", name: "Notes", version: "1", defaultModel: "qwen3:8b", toolAllowlist: [search.id] },
  input: "Find the project plan", tools: [search],
});
for await (const event of run.events()) console.log(event);
console.log(await run.result());
```

Use the `policy` callback for application authorization and `approve` for an
explicit user decision. Both return typed decisions; missing or throwing
handlers fail closed for state-changing tools and approvals. Call
`run.cancel()` for cooperative cancellation and `client.close()` at shutdown.
`runtimeArgs` exists for deterministic test harnesses, not normal distribution.

Use `await client.health()` and `await client.listModels()` for typed provider
inspection before a run. Those calls are separate JSONL commands: they do not
start an agent loop, create a run ID, or request a host tool callback.
