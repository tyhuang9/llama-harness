import asyncio
import os
import sys
import unittest
from pathlib import Path

from llama_harness import HarnessClient, ProviderHealth, ProviderModel, RuntimeProtocolError, Tool


class RuntimeHandshakeTests(unittest.IsolatedAsyncioTestCase):
    def fake_runtime_args(self, mode: str) -> tuple[str, list[str]]:
        return sys.executable, ["-u", str(Path(__file__).with_name("fake_runtime.py")), mode]

    async def test_negotiates_current_protocol_and_exposes_metadata_only_event_shapes(self) -> None:
        path, args = self.fake_runtime_args("modern")
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args) as client:
            self.assertEqual(client.protocol_version, "1.1")
            advanced = Tool(id="metadata", name="Metadata", description="Metadata fixture", arguments_schema={}, execute=lambda **_: {}, output_schema={"type": "object"}, parallel_safe=True, concurrency_key="fixture", cancellation_safety="guaranteed", expected_latency_ms=1, allowed_callers=("programmatic",), issue_safety="guaranteed", execution_location="local_private", network_egress="prohibited")
            run = await client.run(agent={"id": "agent", "name": "Agent", "version": "1", "default_model": "mock"}, input="test", tools=[advanced], strategy="programmatic")
            self.assertEqual((await run.result())["status"], "completed")

    async def test_legacy_fallback_rejects_advanced_modes_before_start(self) -> None:
        path, args = self.fake_runtime_args("legacy")
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args) as client:
            self.assertEqual(client.protocol_version, "1.0")
            with self.assertRaises(RuntimeProtocolError):
                await client.run(agent={"id": "agent", "name": "Agent", "version": "1", "default_model": "mock"}, input="test", strategy="declarative_plan")
            run = await client.run(agent={"id": "agent", "name": "Agent", "version": "1", "default_model": "mock"}, input="test", strategy="direct")
            self.assertEqual((await run.result())["status"], "completed")

    async def test_rejects_incompatible_major_drift_and_structured_protocol_error(self) -> None:
        path, args = self.fake_runtime_args("incompatible")
        with self.assertRaises(RuntimeProtocolError):
            await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args)
        path, args = self.fake_runtime_args("version_mismatch")
        with self.assertRaisesRegex(RuntimeProtocolError, "Runtime version mismatch"):
            await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args)
        path, args = self.fake_runtime_args("drift")
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args) as client:
            with self.assertRaises(RuntimeProtocolError):
                await client.run(agent={"id": "agent", "name": "Agent", "version": "1", "default_model": "mock"}, input="test")
        path, args = self.fake_runtime_args("protocol_error")
        async with await HarnessClient.start(provider={"kind": "ollama"}, runtime_path=path, runtime_args=args) as client:
            with self.assertRaisesRegex(RuntimeProtocolError, "test protocol failure") as caught:
                await client.run(agent={"id": "agent", "name": "Agent", "version": "1", "default_model": "mock"}, input="test")
            self.assertEqual(caught.exception.code, "invalid_state")
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
            models = await client.list_models()
            self.assertEqual([(model.id, model.supports_tools, model.supports_streaming, model.supports_structured_output) for model in models], [("mock-model", True, False, True)])
            self.assertIsNotNone(models[0].capabilities)
