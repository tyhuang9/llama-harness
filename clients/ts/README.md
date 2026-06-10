# @llama-harness/client

Small TypeScript client for local apps that call llama-harness over HTTP.

```ts
import { LlamaHarnessClient } from "@llama-harness/client";

const harness = new LlamaHarnessClient({ baseUrl: "http://127.0.0.1:8787" });

const health = await harness.health();
const models = await harness.listModels();
const response = await harness.chat({
  prompt: "Extract action items from this note.",
  source_app: "note",
});
```

Streaming uses the harness `POST /api/chat/stream` SSE endpoint:

```ts
for await (const event of harness.streamChat({ prompt: "Write a short summary." })) {
  if (event.event === "token") {
    process.stdout.write(JSON.parse(event.data).content);
  }
}
```

