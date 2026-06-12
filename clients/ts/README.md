# @llama-harness/client

Small TypeScript client for local apps that call llama-harness over HTTP.

```ts
import { LlamaHarnessClient } from "@llama-harness/client";

const harness = new LlamaHarnessClient({ baseUrl: "http://127.0.0.1:8787" });

const health = await harness.health();
const models = await harness.listModels();
const response = await harness.chat({
  prompt: "Extract action items from this note.",
  instructions: "Return only actionable checklist items.",
  source_app: "note",
});
```

Provider routes:

```ts
const providers = await harness.listProviders();

const litellm = await harness.testLiteLLMProvider({
  model: "openai:gpt-4o",
  message: "Say hello from llama-harness.",
});

await harness.generateLiteLLMConfig("litellm.config.yaml");

const cloudResponse = await harness.chat({
  provider: "litellm",
  model: "openai:gpt-4o",
  messages: [{ role: "user", content: "Write a quick project summary." }],
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
