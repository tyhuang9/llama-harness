"""Deterministic protocol fixture used by the Python SDK negotiation tests."""

import json
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "legacy"
selected = "2.0" if mode == "incompatible" else "1.1" if mode in {"modern", "drift", "protocol_error", "version_mismatch"} else "1.0"

for line in sys.stdin:
    message = json.loads(line)
    kind = message["type"]
    if kind == "client_hello":
        if message["payload"].get("sdk", {}).get("version") != "0.2.0":
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
        if mode == "modern" and (request.get("strategy") != "programmatic" or not first_tool or first_tool.get("output_schema") is None or first_tool.get("parallel_safe") is not True or first_tool.get("execution_location") != "local_private"):
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "1.1 metadata was not serialized", "retryable": False}}
            print(json.dumps(response), flush=True)
            continue
        if mode == "legacy" and first_tool and ("output_schema" in first_tool or "parallel_safe" in first_tool or "strategy" in request):
            response = {"protocol_version": selected, "request_id": message["request_id"], "type": "protocol_error", "payload": {"code": "invalid_message", "message": "legacy request included 1.1 metadata", "retryable": False}}
            print(json.dumps(response), flush=True)
            continue
        response = {"protocol_version": selected, "request_id": message["request_id"], "run_id": "test-run", "type": "command_acknowledged", "payload": {"command": "start_run"}}
        print(json.dumps(response), flush=True)
        if mode == "drift":
            response = {"protocol_version": "1.0", "request_id": "drift", "run_id": "test-run", "type": "run_event", "payload": {"trace_id": "trace", "sequence": 1, "timestamp_ms": 1, "event": {"type": "model_responded", "call_number": 1}}}
        else:
            response = {"protocol_version": selected, "request_id": "complete", "run_id": "test-run", "type": "run_completed", "payload": {"result": {"status": "completed", "final_output": "ok", "model": "mock", "trace_id": "trace", "duration_ms": 1}}}
    elif kind == "shutdown":
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "command_acknowledged", "payload": {"command": "shutdown"}}
    else:
        response = {"protocol_version": selected, "request_id": message["request_id"], "type": "command_acknowledged", "payload": {"command": kind}}
    print(json.dumps(response), flush=True)
