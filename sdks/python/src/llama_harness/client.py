"""Protocol client and child-process lifecycle for llama-harness."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import platform
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Json = None | bool | int | float | str | list["Json"] | dict[str, "Json"]
PolicyOutcome = Literal["allow", "deny", "require_approval"]

class HarnessError(Exception): pass
class RuntimeUnavailableError(HarnessError): pass
class RuntimeProtocolError(HarnessError): pass
class RuntimeExitedError(HarnessError): pass
class RunCancelledError(HarnessError): pass

@dataclass(frozen=True)
class ToolContext:
    run_id: str
    trace_id: str
    call_id: str
    deadline_ms: int | None = None

@dataclass(frozen=True)
class Tool:
    id: str
    description: str
    arguments_schema: dict[str, Json]
    execute: Callable[..., Any]
    name: str | None = None
    risk: Literal["low", "medium", "high"] = "low"
    read_only: bool = True
    idempotent: bool = True

@dataclass(frozen=True)
class PolicyRequest:
    run_id: str
    trace_id: str
    call_id: str
    tool: Tool
    arguments: Json
    deadline_ms: int | None = None

ApprovalRequest = PolicyRequest
PolicyCallback = Callable[[PolicyRequest], Awaitable[dict[str, str]] | dict[str, str]]
ApprovalCallback = Callable[[ApprovalRequest], Awaitable[dict[str, Any]] | dict[str, Any]]

def tool(*, id: str, description: str, arguments_schema: dict[str, Json], name: str | None = None,
         risk: Literal["low", "medium", "high"] = "low", read_only: bool = True,
         idempotent: bool = True) -> Callable[[Callable[..., Any]], Tool]:
    def decorate(function: Callable[..., Any]) -> Tool:
        return Tool(id=id, name=name or function.__name__, description=description,
                    arguments_schema=arguments_schema, risk=risk, read_only=read_only,
                    idempotent=idempotent, execute=function)
    return decorate

@dataclass
class _RunState:
    run_id: str
    tools: dict[str, Tool]
    policy: PolicyCallback | None
    approve: ApprovalCallback | None
    events: asyncio.Queue[dict[str, Any] | BaseException | None] = field(default_factory=asyncio.Queue)
    result: asyncio.Future[dict[str, Any]] | None = None

class HarnessRun:
    def __init__(self, client: "HarnessClient", state: _RunState) -> None:
        self._client, self._state = client, state

    @property
    def id(self) -> str: return self._state.run_id

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self._state.events.get()
            if event is None: return
            if isinstance(event, BaseException): raise event
            yield event

    async def result(self) -> dict[str, Any]:
        assert self._state.result is not None
        return await self._state.result

    async def cancel(self, reason: str = "cancelled by Python SDK host") -> None:
        await self._client._send("cancel_run", {"reason": reason}, run_id=self.id)

class HarnessClient:
    def __init__(self, process: asyncio.subprocess.Process, *, provider: Mapping[str, Any], stderr: Callable[[str], None] | None) -> None:
        self._process, self._provider, self._stderr = process, provider, stderr
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._runs: dict[str, _RunState] = {}
        self._buffered: dict[str, list[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._closed = False
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

    @classmethod
    async def start(cls, *, provider: Mapping[str, Any], runtime_path: str | os.PathLike[str] | None = None,
                    runtime_args: list[str] | None = None, on_stderr: Callable[[str], None] | None = None) -> "HarnessClient":
        path = resolve_runtime_path(runtime_path)
        try:
            process = await asyncio.create_subprocess_exec(path, *(runtime_args or []), stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, limit=256 * 1024)
        except OSError as error:
            raise RuntimeUnavailableError(f"Could not start {path}: {error}") from error
        client = cls(process, provider=provider, stderr=on_stderr)
        response = await client._request("client_hello", {"sdk": {"name": "llama-harness-python", "version": "0.1.0"}, "capabilities": ["async_callbacks"]})
        if response["type"] != "runtime_hello":
            await client.close()
            raise RuntimeProtocolError(f"Expected runtime_hello, received {response['type']}")
        return client

    async def __aenter__(self) -> "HarnessClient": return self
    async def __aexit__(self, *_: object) -> None: await self.close()

    async def close(self) -> None:
        if self._closed: return
        try: await self._request("shutdown", {})
        except HarnessError: self._process.kill()
        finally:
            self._closed = True
            if self._process.stdin: self._process.stdin.close()
            await self._process.wait()
            self._reader_task.cancel(); self._stderr_task.cancel()

    async def run(self, *, agent: Mapping[str, Any], input: str, tools: list[Tool] | None = None,
                  policy: PolicyCallback | None = None, approve: ApprovalCallback | None = None,
                  application_context: Mapping[str, Json] | None = None, metadata: Mapping[str, Json] | None = None,
                  evaluation: Mapping[str, Json] | None = None, model: str | None = None) -> HarnessRun:
        selected_tools = tools or []
        response = await self._request("start_run", {"request": {"provider": _provider(self._provider), "agent": _agent(agent), "input": input,
            "tools": [_tool(value) for value in selected_tools], "application_context": application_context or {}, "metadata": metadata or {},
            "evaluation": evaluation or {}, "overrides": {"model": model, "generation": {}}}})
        run_id = response.get("run_id")
        if response["type"] != "command_acknowledged" or not isinstance(run_id, str):
            raise RuntimeProtocolError("Runtime did not acknowledge start_run with a run ID")
        state = _RunState(run_id, {value.id: value for value in selected_tools}, policy, approve, result=asyncio.get_running_loop().create_future())
        self._runs[run_id] = state
        for envelope in self._buffered.pop(run_id, []): self._dispatch_run(state, envelope)
        return HarnessRun(self, state)

    async def _request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4()); future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future(); self._pending[request_id] = future
        await self._send(kind, payload, request_id=request_id)
        return await future

    async def _send(self, kind: str, payload: dict[str, Any], *, request_id: str | None = None, run_id: str | None = None) -> None:
        if self._closed or not self._process.stdin: raise RuntimeExitedError("Harness client is closed")
        envelope = {"protocol_version": "1.0", "request_id": request_id or str(uuid.uuid4()), "type": kind, "payload": payload}
        if run_id: envelope["run_id"] = run_id
        async with self._write_lock:
            self._process.stdin.write((json.dumps(envelope, separators=(",", ":")) + "\n").encode())
            await self._process.stdin.drain()

    async def _read_stdout(self) -> None:
        assert self._process.stdout
        try:
            while line := await self._process.stdout.readline(): self._receive(_envelope(line))
            if not self._closed: self._fail_all(RuntimeExitedError("Runtime stdout closed"))
        except asyncio.CancelledError: pass
        except BaseException as error:
            self._fail_all(RuntimeProtocolError(f"Runtime stdout corruption: {error}"))
            self._process.kill()

    async def _read_stderr(self) -> None:
        assert self._process.stderr
        try:
            while line := await self._process.stderr.readline():
                if self._stderr: self._stderr(line.decode(errors="replace").rstrip())
        except asyncio.CancelledError: pass

    def _receive(self, envelope: dict[str, Any]) -> None:
        pending = self._pending.pop(envelope["request_id"], None)
        if pending and envelope["type"] in {"runtime_hello", "command_acknowledged", "protocol_error", "pong"}:
            pending.set_result(envelope); return
        run_id = envelope.get("run_id")
        if not isinstance(run_id, str): return
        state = self._runs.get(run_id)
        if not state: self._buffered.setdefault(run_id, []).append(envelope); return
        self._dispatch_run(state, envelope)

    def _dispatch_run(self, state: _RunState, envelope: dict[str, Any]) -> None:
        kind, payload = envelope["type"], envelope["payload"]
        if kind == "run_event": state.events.put_nowait(payload)
        elif kind == "tool_execution_requested": asyncio.create_task(self._tool_callback(state, payload))
        elif kind == "policy_decision_requested": asyncio.create_task(self._policy_callback(state, payload))
        elif kind == "approval_requested": asyncio.create_task(self._approval_callback(state, payload))
        elif kind == "run_completed": state.events.put_nowait(None); state.result and state.result.set_result(_result(payload["result"])); self._runs.pop(state.run_id, None)
        elif kind == "run_cancelled": self._finish_error(state, RunCancelledError(str(payload.get("reason", "Run cancelled"))))
        elif kind == "run_failed": self._finish_error(state, HarnessError(str(payload.get("error", {}).get("message", "Run failed"))))

    def _finish_error(self, state: _RunState, error: BaseException) -> None:
        state.events.put_nowait(error); state.events.put_nowait(None)
        if state.result and not state.result.done(): state.result.set_exception(error)
        self._runs.pop(state.run_id, None)

    async def _tool_callback(self, state: _RunState, payload: dict[str, Any]) -> None:
        callback_id = payload["callback_id"]
        try:
            tool_value = state.tools[str(payload["tool"]["id"])]
            output = await _call_tool(tool_value, payload["arguments"], ToolContext(state.run_id, str(payload["trace_id"]), str(payload["call_id"]), payload.get("deadline_ms")))
            result = {"ok": True, "output": output}
        except BaseException as error: result = {"ok": False, "output": None, "error": str(error)}
        await self._send("tool_result", {"callback_id": callback_id, "result": result}, run_id=state.run_id)

    async def _policy_callback(self, state: _RunState, payload: dict[str, Any]) -> None:
        request = _callback_request(state, payload)
        try: decision = await _maybe_await(state.policy(request)) if state.policy else ({"outcome": "allow", "reason": "read-only tool allowed by SDK default policy"} if request.tool.read_only else {"outcome": "deny", "reason": "state-changing tool requires an explicit policy"})
        except BaseException as error: decision = {"outcome": "deny", "reason": f"SDK policy handler failed: {error}"}
        await self._send("policy_decision", {"callback_id": payload["callback_id"], "decision": decision}, run_id=state.run_id)

    async def _approval_callback(self, state: _RunState, payload: dict[str, Any]) -> None:
        request = _callback_request(state, payload)
        try: decision = await _maybe_await(state.approve(request)) if state.approve else {"granted": False, "reason": "no SDK approval handler configured"}
        except BaseException as error: decision = {"granted": False, "reason": f"SDK approval handler failed: {error}"}
        await self._send("approval_decision", {"callback_id": payload["callback_id"], **decision}, run_id=state.run_id)

    def _fail_all(self, error: BaseException) -> None:
        for future in self._pending.values():
            if not future.done(): future.set_exception(error)
        self._pending.clear()
        for state in list(self._runs.values()): self._finish_error(state, error)

def resolve_runtime_path(explicit: str | os.PathLike[str] | None = None) -> str:
    if explicit: return os.fspath(explicit)
    if environment := os.environ.get("LLAMA_HARNESS_RUNTIME_PATH"): return environment
    binary = "llama-harness-runtime.exe" if os.name == "nt" else "llama-harness-runtime"
    packaged = Path(__file__).with_name("runtime") / binary
    if packaged.is_file(): return str(packaged)
    raise RuntimeUnavailableError(f"No packaged runtime for {platform.system().lower()}-{platform.machine().lower()}. Set LLAMA_HARNESS_RUNTIME_PATH explicitly.")

async def _maybe_await(value: Any) -> Any: return await value if inspect.isawaitable(value) else value
async def _call_tool(value: Tool, arguments: Json, context: ToolContext) -> Json:
    result = value.execute(**arguments, context=context) if isinstance(arguments, dict) else value.execute(arguments, context=context)
    return await _maybe_await(result)
def _envelope(line: bytes) -> dict[str, Any]:
    value = json.loads(line)
    if str(value.get("protocol_version", "")).split(".")[0] != "1" or not value.get("request_id") or not value.get("type") or not isinstance(value.get("payload"), dict): raise RuntimeProtocolError("Malformed protocol envelope")
    return value
def _tool(value: Tool) -> dict[str, Json]: return {"id": value.id, "name": value.name or value.id, "description": value.description, "arguments_schema": value.arguments_schema, "risk": value.risk, "read_only": value.read_only, "idempotent": value.idempotent}
def _provider(value: Mapping[str, Any]) -> dict[str, Json]:
    if value.get("kind") != "ollama": raise HarnessError("Only the ollama provider is supported by sidecar v1")
    return {"kind": "ollama", "base_url": str(value.get("base_url", value.get("baseUrl", "http://127.0.0.1:11434")))}
def _agent(value: Mapping[str, Any]) -> dict[str, Json]:
    return {"id": str(value["id"]), "name": str(value["name"]), "version": str(value["version"]), "system_instructions": str(value.get("instructions", "")), "default_model": str(value.get("default_model", value.get("defaultModel"))), "tool_allowlist": list(value.get("tool_allowlist", value.get("toolAllowlist", []))), "limits": value.get("limits", {}), "generation": value.get("generation", {}), "metadata": value.get("metadata", {})}
def _result(value: Mapping[str, Any]) -> dict[str, Any]: return {**value, "final_output": value.get("final_output"), "trace_id": value.get("trace_id")}
def _callback_request(state: _RunState, payload: Mapping[str, Any]) -> PolicyRequest:
    wire = payload["tool"]; value = Tool(id=str(wire["id"]), name=str(wire["name"]), description=str(wire["description"]), arguments_schema=wire["arguments_schema"], risk=wire["risk"], read_only=wire["read_only"], idempotent=wire["idempotent"], execute=lambda: None)
    return PolicyRequest(state.run_id, str(payload["trace_id"]), str(payload["call_id"]), value, payload["arguments"], payload.get("deadline_ms"))
