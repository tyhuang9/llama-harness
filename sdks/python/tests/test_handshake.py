import asyncio
import os
import unittest
from pathlib import Path

from llama_harness import HarnessClient


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
