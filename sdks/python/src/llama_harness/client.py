"""Protocol client and child-process lifecycle for llama-harness."""

from __future__ import annotations

import asyncio
import inspect
from importlib.metadata import PackageNotFoundError, version as distribution_version
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
RunStrategy = Literal["adaptive", "direct", "declarative_plan", "programmatic"]
CancellationSafety = Literal["unknown", "cooperative", "guaranteed"]
ToolCaller = Literal["direct", "declarative_plan", "programmatic", "speculative"]
SpeculationPolicy = Literal["disabled", "enabled"]
IssueSafety = Literal["unknown", "guaranteed"]
ExecutionLocation = Literal["unknown", "local_private", "remote"]
NetworkEgress = Literal["unknown", "prohibited", "permitted"]
OFFERED_PROTOCOL_VERSION = "1.1"
try:
    SDK_VERSION = distribution_version("llama-harness")
except PackageNotFoundError:
    # Editable source checkouts are not installed distributions.
    SDK_VERSION = "0.2.0"

class HarnessError(Exception): pass
class RuntimeUnavailableError(HarnessError): pass
class RuntimeProtocolError(HarnessError):
    def __init__(self, message: str, code: str | None = None, retryable: bool | None = None) -> None:
        super().__init__(message)
        self.code, self.retryable = code, retryable
class RuntimeExitedError(HarnessError): pass
class RunCancelledError(HarnessError): pass

@dataclass(frozen=True)
class ProviderHealth:
    healthy: bool
    detail: str | None = None

@dataclass(frozen=True)
class ProviderCapabilityLimits:
    max_tools: int | None = None
    max_tool_schema_bytes: int | None = None
    max_parallel_tool_calls: int | None = None
    max_streamed_argument_bytes: int | None = None
    max_streamed_tool_calls: int | None = None
    max_plan_bytes: int | None = None
    max_plan_nodes: int | None = None
    max_program_bytes: int | None = None

@dataclass(frozen=True)
class ProviderModelCapabilities:
    supports_tools: bool
    supports_streaming: bool
    supports_structured_output: bool
    supports_strict_tool_schemas: bool = False
    supports_streaming_tool_arguments: bool = False
    supports_parallel_tool_calls: bool = False
    supports_structured_plans: bool = False
    supports_programmatic_calling: bool = False
    programmatic_conformance: Literal["strict_json_ast_v1"] | None = None
    limits: ProviderCapabilityLimits = field(default_factory=ProviderCapabilityLimits)

@dataclass(frozen=True)
class ProviderModel:
    id: str
    supports_tools: bool
    supports_streaming: bool
    supports_structured_output: bool
    capabilities: ProviderModelCapabilities | None = None

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
    output_schema: dict[str, Json] | None = None
    parallel_safe: bool = False
    concurrency_key: str | None = None
    cancellation_safety: CancellationSafety = "unknown"
    expected_latency_ms: int | None = None
    allowed_callers: tuple[ToolCaller, ...] = ("direct",)
    speculation_policy: SpeculationPolicy = "disabled"
    issue_safety: IssueSafety = "unknown"
    execution_location: ExecutionLocation = "unknown"
    network_egress: NetworkEgress = "unknown"

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
         idempotent: bool = True, output_schema: dict[str, Json] | None = None,
         parallel_safe: bool = False, concurrency_key: str | None = None,
         cancellation_safety: CancellationSafety = "unknown", expected_latency_ms: int | None = None,
         allowed_callers: tuple[ToolCaller, ...] = ("direct",), speculation_policy: SpeculationPolicy = "disabled",
         issue_safety: IssueSafety = "unknown", execution_location: ExecutionLocation = "unknown",
         network_egress: NetworkEgress = "unknown") -> Callable[[Callable[..., Any]], Tool]:
    def decorate(function: Callable[..., Any]) -> Tool:
        return Tool(id=id, name=name or function.__name__, description=description,
                    arguments_schema=arguments_schema, risk=risk, read_only=read_only,
                    idempotent=idempotent, output_schema=output_schema, parallel_safe=parallel_safe,
                    concurrency_key=concurrency_key, cancellation_safety=cancellation_safety,
                    expected_latency_ms=expected_latency_ms, allowed_callers=allowed_callers,
                    speculation_policy=speculation_policy, issue_safety=issue_safety,
                    execution_location=execution_location, network_egress=network_egress, execute=function)
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
        self._negotiated_protocol_version: Literal["1.0", "1.1"] | None = None
        self._terminal_error: BaseException | None = None
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
        try:
            response = await client._request("client_hello", {"sdk": {"name": "llama-harness-python", "version": SDK_VERSION}, "capabilities": ["async_callbacks"]})
            if response["type"] != "runtime_hello":
                raise RuntimeProtocolError(f"Expected runtime_hello, received {response['type']}")
            client._negotiated_protocol_version = _selected_protocol_version(str(response["protocol_version"]))
            runtime_version = response["payload"].get("runtime_version")
            if runtime_version != SDK_VERSION:
                raise RuntimeProtocolError(f"Runtime version mismatch: SDK {SDK_VERSION}, runtime {runtime_version}")
            return client
        except BaseException:
            await client._abort_startup()
            raise

    @property
    def protocol_version(self) -> str | None:
        """Exact version selected by the runtime hello envelope."""
        return self._negotiated_protocol_version

    async def __aenter__(self) -> "HarnessClient": return self
    async def __aexit__(self, *_: object) -> None: await self.close()

    async def close(self) -> None:
        if self._closed: return
        try: await self._request("shutdown", {})
        except (HarnessError, OSError, ConnectionError):
            if self._process.returncode is None: self._process.kill()
        finally:
            self._closed = True
            if self._process.stdin: self._process.stdin.close()
            await self._process.wait()
            self._reader_task.cancel(); self._stderr_task.cancel()

    async def _abort_startup(self) -> None:
        """Bound cleanup when no trustworthy protocol session was established."""
        self._closed = True
        if self._process.stdin:
            self._process.stdin.close()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            if self._process.returncode is None:
                self._process.kill()
            await self._process.wait()
        finally:
            self._reader_task.cancel()
            self._stderr_task.cancel()
            await asyncio.gather(self._reader_task, self._stderr_task, return_exceptions=True)

    async def run(self, *, agent: Mapping[str, Any], input: str, tools: list[Tool] | None = None,
                  policy: PolicyCallback | None = None, approve: ApprovalCallback | None = None,
                  application_context: Mapping[str, Json] | None = None, metadata: Mapping[str, Json] | None = None,
                  evaluation: Mapping[str, Json] | None = None, model: str | None = None,
                  generation: Mapping[str, Json] | None = None, strategy: RunStrategy | None = None) -> HarnessRun:
        if self._terminal_error: raise self._terminal_error
        if strategy in {"declarative_plan", "programmatic"} and self._negotiated_protocol_version != "1.1":
            raise RuntimeProtocolError(f"{strategy} requires negotiated protocol version 1.1")
        selected_tools = tools or []
        request: dict[str, Any] = {"provider": _provider(self._provider), "agent": _agent(agent), "input": input,
            "tools": [_tool(value, self._negotiated_protocol_version == "1.1") for value in selected_tools], "application_context": application_context or {}, "metadata": metadata or {},
            "evaluation": evaluation or {}, "overrides": {"model": model, "generation": generation or {}}}
        if self._negotiated_protocol_version == "1.1" and strategy is not None:
            request["strategy"] = strategy
        response = await self._request("start_run", {"request": request})
        if self._terminal_error: raise self._terminal_error
        run_id = response.get("run_id")
        if response["type"] != "command_acknowledged" or not isinstance(run_id, str):
            raise RuntimeProtocolError("Runtime did not acknowledge start_run with a run ID")
        state = _RunState(run_id, {value.id: value for value in selected_tools}, policy, approve, result=asyncio.get_running_loop().create_future())
        self._runs[run_id] = state
        for envelope in self._buffered.pop(run_id, []): self._dispatch_run(state, envelope)
        return HarnessRun(self, state)

    async def health(self) -> ProviderHealth:
        response = await self._request("get_provider_health", {"provider": _provider(self._provider)})
        if response["type"] != "provider_health":
            raise RuntimeProtocolError(f"Expected provider_health, received {response['type']}")
        payload = response["payload"]
        return ProviderHealth(bool(payload["healthy"]), str(payload["detail"]) if payload.get("detail") is not None else None)

    async def list_models(self) -> list[ProviderModel]:
        response = await self._request("get_model_inventory", {"provider": _provider(self._provider)})
        if response["type"] != "model_inventory":
            raise RuntimeProtocolError(f"Expected model_inventory, received {response['type']}")
        models = response["payload"].get("models")
        if not isinstance(models, list):
            raise RuntimeProtocolError("Model inventory did not contain models")
        return [_provider_model(value) for value in models]

    async def _request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = str(uuid.uuid4()); future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future(); self._pending[request_id] = future
        await self._send(kind, payload, request_id=request_id)
        response = await future
        if response["type"] == "protocol_error":
            raise _protocol_error(response["payload"])
        return response

    async def _send(self, kind: str, payload: dict[str, Any], *, request_id: str | None = None, run_id: str | None = None) -> None:
        if self._closed or not self._process.stdin: raise RuntimeExitedError("Harness client is closed")
        envelope = {"protocol_version": self._negotiated_protocol_version or OFFERED_PROTOCOL_VERSION, "request_id": request_id or str(uuid.uuid4()), "type": kind, "payload": payload}
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
        if self._negotiated_protocol_version and envelope["protocol_version"] != self._negotiated_protocol_version:
            self._terminal_error = RuntimeProtocolError(f"Runtime protocol version drift: expected {self._negotiated_protocol_version}, received {envelope['protocol_version']}")
            self._fail_all(self._terminal_error)
            self._process.kill()
            return
        pending = self._pending.pop(envelope["request_id"], None)
        if pending and envelope["type"] in {"runtime_hello", "command_acknowledged", "protocol_error", "pong", "provider_health", "model_inventory"}:
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
        elif kind == "protocol_error": self._finish_error(state, _protocol_error(payload))

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
def _selected_protocol_version(value: str) -> Literal["1.0", "1.1"]:
    if value in {"1.0", "1.1"}: return value
    raise RuntimeProtocolError(f"Unsupported runtime protocol version {value}", "incompatible_version", False)
def _protocol_error(payload: Mapping[str, Any]) -> RuntimeProtocolError:
    return RuntimeProtocolError(str(payload.get("message", "Runtime protocol error")), str(payload["code"]) if payload.get("code") is not None else None, bool(payload["retryable"]) if payload.get("retryable") is not None else None)
def _wire_bool(value: Any, default: bool = False) -> bool: return value if isinstance(value, bool) else default
def _wire_enum(value: Any, allowed: set[str], default: str) -> str: return value if isinstance(value, str) and value in allowed else default
def _wire_callers(value: Any) -> tuple[ToolCaller, ...]:
    if not isinstance(value, list): return ("direct",)
    return tuple(entry for entry in value if entry in {"direct", "declarative_plan", "programmatic", "speculative"})
def _tool(value: Tool, include_v11_metadata: bool) -> dict[str, Json]:
    result: dict[str, Json] = {"id": value.id, "name": value.name or value.id, "description": value.description, "arguments_schema": value.arguments_schema, "risk": value.risk, "read_only": value.read_only, "idempotent": value.idempotent}
    if include_v11_metadata:
        result.update({"parallel_safe": value.parallel_safe, "cancellation_safety": value.cancellation_safety, "allowed_callers": list(value.allowed_callers), "speculation_policy": value.speculation_policy, "issue_safety": value.issue_safety, "execution_location": value.execution_location, "network_egress": value.network_egress})
        if value.output_schema is not None: result["output_schema"] = value.output_schema
        if value.concurrency_key is not None: result["concurrency_key"] = value.concurrency_key
        if value.expected_latency_ms is not None: result["expected_latency_ms"] = value.expected_latency_ms
    return result
def _provider_model(value: Mapping[str, Any]) -> ProviderModel:
    capabilities = value.get("capabilities")
    if not isinstance(capabilities, Mapping): raise RuntimeProtocolError("Model inventory entry did not contain capabilities")
    limits = capabilities.get("limits")
    safe_limits = ProviderCapabilityLimits(**{key: item for key, item in (limits.items() if isinstance(limits, Mapping) else []) if key in ProviderCapabilityLimits.__dataclass_fields__ and isinstance(item, int)})
    metadata = ProviderModelCapabilities(bool(capabilities.get("supports_tools")), bool(capabilities.get("supports_streaming")), bool(capabilities.get("supports_structured_output")), bool(capabilities.get("supports_strict_tool_schemas")), bool(capabilities.get("supports_streaming_tool_arguments")), bool(capabilities.get("supports_parallel_tool_calls")), bool(capabilities.get("supports_structured_plans")), bool(capabilities.get("supports_programmatic_calling")), capabilities.get("programmatic_conformance") if capabilities.get("programmatic_conformance") == "strict_json_ast_v1" else None, safe_limits)
    return ProviderModel(str(value["id"]), metadata.supports_tools, metadata.supports_streaming, metadata.supports_structured_output, metadata)
def _provider(value: Mapping[str, Any]) -> dict[str, Json]:
    if value.get("kind") != "ollama": raise HarnessError("Only the ollama provider is supported by sidecar v1")
    return {"kind": "ollama", "base_url": str(value.get("base_url", value.get("baseUrl", "http://127.0.0.1:11434")))}
def _agent(value: Mapping[str, Any]) -> dict[str, Json]:
    return {"id": str(value["id"]), "name": str(value["name"]), "version": str(value["version"]), "system_instructions": str(value.get("instructions", "")), "default_model": str(value.get("default_model", value.get("defaultModel"))), "tool_allowlist": list(value.get("tool_allowlist", value.get("toolAllowlist", []))), "limits": value.get("limits", {}), "generation": value.get("generation", {}), "metadata": value.get("metadata", {})}
def _result(value: Mapping[str, Any]) -> dict[str, Any]: return {**value, "final_output": value.get("final_output"), "trace_id": value.get("trace_id")}
def _callback_request(state: _RunState, payload: Mapping[str, Any]) -> PolicyRequest:
    wire = payload["tool"]; value = Tool(id=str(wire["id"]), name=str(wire["name"]), description=str(wire["description"]), arguments_schema=wire["arguments_schema"], risk=_wire_enum(wire.get("risk"), {"low", "medium", "high"}, "high"), read_only=_wire_bool(wire.get("read_only")), idempotent=_wire_bool(wire.get("idempotent")), output_schema=wire.get("output_schema"), parallel_safe=_wire_bool(wire.get("parallel_safe")), concurrency_key=wire.get("concurrency_key") if isinstance(wire.get("concurrency_key"), str) else None, cancellation_safety=_wire_enum(wire.get("cancellation_safety"), {"unknown", "cooperative", "guaranteed"}, "unknown"), expected_latency_ms=wire.get("expected_latency_ms") if isinstance(wire.get("expected_latency_ms"), int) else None, allowed_callers=_wire_callers(wire.get("allowed_callers")), speculation_policy=_wire_enum(wire.get("speculation_policy"), {"disabled", "enabled"}, "disabled"), issue_safety=_wire_enum(wire.get("issue_safety"), {"unknown", "guaranteed"}, "unknown"), execution_location=_wire_enum(wire.get("execution_location"), {"unknown", "local_private", "remote"}, "unknown"), network_egress=_wire_enum(wire.get("network_egress"), {"unknown", "prohibited", "permitted"}, "unknown"), execute=lambda: None)
    return PolicyRequest(state.run_id, str(payload["trace_id"]), str(payload["call_id"]), value, payload["arguments"], payload.get("deadline_ms"))
