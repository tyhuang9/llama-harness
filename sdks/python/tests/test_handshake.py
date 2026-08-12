import asyncio
import os
import unittest
from pathlib import Path

from llama_harness import HarnessClient, ProviderHealth, ProviderModel, Tool


class RuntimeHandshakeTests(unittest.IsolatedAsyncioTestCase):
    async def test_workspace_runtime_handshake(self) -> None:
        runtime = Path(__file__).resolve().parents[3] / "target" / "debug" / "llama-harness-runtime.exe"
        if not runtime.is_file():
            self.skipTest("workspace runtime has not been built")
        previous = os.environ.get("LLAMA_HARNESS_RUNTIME_PATH")
        os.environ["LLAMA_HARNESS_RUNTIME_PATH"] = str(runtime)
        try:
            async with await HarnessClient.start(provider={"kind": "ollama"}) as client:
                self.assertIsNotNone(client)
        finally:
            if previous is None:
                del os.environ["LLAMA_HARNESS_RUNTIME_PATH"]
            else:
                os.environ["LLAMA_HARNESS_RUNTIME_PATH"] = previous

    async def test_workspace_scripted_runtime_completes_host_callbacks(self) -> None:
        runtime = Path(__file__).resolve().parents[3] / "target" / "debug" / "llama-harness-scripted-runtime.exe"
        if not runtime.is_file():
            self.skipTest("workspace scripted runtime has not been built")
        calls = {"policy": 0, "approval": 0, "tool": 0}

        async def search(query: str, *, context: object) -> dict[str, str]:
            calls["tool"] += 1
            self.assertEqual(query, "harness")
            self.assertIsNotNone(context)
            return {"found": query}

        async def policy(_: object) -> dict[str, str]:
            calls["policy"] += 1
            return {"outcome": "require_approval", "reason": "test approval routing"}

        async def approve(_: object) -> dict[str, object]:
            calls["approval"] += 1
            return {"granted": True, "reason": "approved by test host"}

        tool = Tool(
            id="notes.search", name="Search notes", description="Search locally owned notes",
            arguments_schema={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
            risk="medium", read_only=False, idempotent=True, execute=search,
        )
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=runtime) as client:
            run = await client.run(
                agent={"id": "scripted", "name": "Scripted", "version": "1", "default_model": "mock", "tool_allowlist": [tool.id]},
                input="find harness", tools=[tool], policy=policy, approve=approve,
            )
            events = [event async for event in run.events()]
            result = await run.result()

        self.assertEqual(result["status"], "completed")
        self.assertIn("host tool callback", result.get("final_output") or "")
        self.assertEqual(calls, {"policy": 1, "approval": 1, "tool": 1})
        self.assertGreaterEqual(len(events), 4)
        self.assertEqual([event["sequence"] for event in events], sorted(event["sequence"] for event in events))

    async def test_workspace_scripted_runtime_exposes_provider_inspection(self) -> None:
        runtime = Path(__file__).resolve().parents[3] / "target" / "debug" / "llama-harness-scripted-runtime.exe"
        if not runtime.is_file():
            self.skipTest("workspace scripted runtime has not been built")
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=runtime) as client:
            self.assertEqual(await client.health(), ProviderHealth(True))
            self.assertEqual(await client.list_models(), [ProviderModel("mock-model", True, False, True)])
