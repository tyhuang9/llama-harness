# Python SDK

`llama_harness` is asyncio-native and starts its Rust child only from
`await HarnessClient.start(...)`. Install the matching reviewed platform wheel
or use `LLAMA_HARNESS_RUNTIME_PATH` with a workspace-built executable in local
development. It never downloads a runtime.

```python
from llama_harness import HarnessClient, tool

@tool(id="notes.search", description="Search local notes", arguments_schema={"type": "object"})
async def search(query: str, *, context):
    return {"matches": await find_local_notes(query)}

async with await HarnessClient.start(provider={"kind": "ollama"}) as client:
    run = await client.run(
        agent={"id": "notes", "name": "Notes", "version": "1", "default_model": "qwen3:8b", "tool_allowlist": [search.id]},
        input="Find the project plan", tools=[search],
    )
    async for event in run.events():
        print(event)
    print(await run.result())
```

`policy` and `approve` may be synchronous or asynchronous. Exceptions fail
closed. Always close the client (the context manager does so), and treat
cancellation as cooperative: it stops subsequent agent work but cannot reverse
a tool's completed external side effect.
