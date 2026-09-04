"""Deterministic protocol fixture used by the Python SDK negotiation tests."""

import json
import sys
from pathlib import Path

mode = sys.argv[1] if len(sys.argv) > 1 else "legacy"
selected = "2.0" if mode == "incompatible" else "1.1" if mode in {"modern", "agent_schema", "drift", "protocol_error", "version_mismatch", "malformed_metadata", "hello_error"} else "1.0"
shutdown_marker = Path(sys.argv[2]) if len(sys.argv) > 2 else None

for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "client_hello":
        if mode == "hello_error":
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_state", "message": "hello rejected", "retryable": False}}
        elif message["payload"].get("sdk", {}).get("version") != "0.2.0":
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "SDK identity version mismatch", "retryable": False}}
        else:
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "runtime_hello", "payload": {"runtime_version": "0.1.0" if mode == "version_mismatch" else "0.2.0", "capabilities": {}, "providers": ["ollama"]}}
    elif message.get("protocol_version") != selected:
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "incompatible_version", "message": "client version drift", "retryable": False}}
    elif kind == "start_run" and mode == "protocol_error":
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_state", "message": "test protocol failure", "retryable": False}}
    elif kind == "start_run":
        request = message["payload"]["request"]
        tools = request.get("tools", [])
        first_tool = tools[0] if tools else None
        agent_schema = request.get("agent", {}).get("output_schema")
        if mode == "modern" and (request.get("strategy") != "programmatic" or not first_tool or first_tool.get("output_schema") is None or first_tool.get("parallel_safe") is not True or first_tool.get("execution_location") != "local_private"):
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "1.1 metadata was not serialized", "retryable": False}}
            print(json.dumps(response), flush=True)
            continue
        if mode == "agent_schema" and agent_schema != {"type": "object", "additionalProperties": False}:
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "agent output schema was not serialized", "retryable": False}}
            print(json.dumps(response), flush=True)
            continue
        if mode == "legacy" and ((first_tool and ("output_schema" in first_tool or "parallel_safe" in first_tool)) or "strategy" in request):
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "legacy request included 1.1 metadata", "retryable": False}}
            print(json.dumps(response), flush=True)
            continue
        response = {"protocol_version": selected, "request_id": message["request_id"], "run_id": "test-run", "type": "command_acknowledged", "payload": {"command": "start_run"}}
        print(json.dumps(response), flush=True)
        if mode == "malformed_metadata":
            response = {"protocol_version": selected, "request_id": "policy-request", "run_id": "test-run", "type": "policy_decision_requested", "payload": {"callback_id": "policy-1", "trace_id": "trace", "call_id": "call-1", "tool": {"id": "notes.search", "name": "Search", "description": "Search", "arguments_schema": {}, "risk": "invalid", "idempotent": "true", "read_only": "true", "parallel_safe": "true", "cancellation_safety": "invalid", "allowed_callers": ["invalid"], "speculation_policy": "invalid", "issue_safety": "invalid", "execution_location": "invalid", "network_egress": "invalid"}, "arguments": {}}}
        elif mode == "drift":
            response = {"protocol_version": "1.0", "request_id": "drift", "run_id": "test-run", "type": "run_event", "payload": {"trace_id": "trace", "sequence": 1, "timestamp_ms": 1, "event": {"type": "model_responded", "call_number": 1}}}
        elif mode == "agent_schema":
            response = {"protocol_version": selected, "request_id": "complete", "run_id": "test-run", "type": "run_completed", "payload": {"result": {"status": "completed", "final_output": "ok", "model": "mock", "trace_id": "trace", "duration_ms": 1}}}
        else:
            response = {"protocol_version": selected, "request_id": "complete", "run_id": "test-run", "type": "run_completed", "payload": {"result": {"status": "completed", "final_output": "ok", "model": "mock", "trace_id": "trace", "duration_ms": 1}}}
    elif kind == "policy_decision" and mode == "malformed_metadata":
        response = {"protocol_version": selected, "request_id": "complete", "run_id": "test-run", "type": "run_completed", "payload": {"result": {"status": "completed", "final_output": "denied", "model": "mock", "trace_id": "trace", "duration_ms": 1}}}
    elif kind == "shutdown":
        if shutdown_marker is not None:
            shutdown_marker.write_text("closed\n", encoding="utf-8")
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "command_acknowledged", "payload": {"command": "shutdown"}}
    else:
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "command_acknowledged", "payload": {"command": kind}}
    print(json.dumps(response), flush=True)

if shutdown_marker is not None and not shutdown_marker.exists():
    shutdown_marker.write_text("closed\n", encoding="utf-8")
