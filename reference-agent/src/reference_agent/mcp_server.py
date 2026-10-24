"""A minimal, runnable MCP (Model Context Protocol) server over stdio.

This is a self-contained JSON-RPC 2.0 server. It fixes the old book bug where
the example returned tools bound to an already-closed session: here the
server owns the tools and stays alive for the whole stdio session.

Run with:  python -m reference_agent.mcp_server
"""

from __future__ import annotations

import json
import math
import sys

from .tools import CalculatorTool

_TOOLS = {"calculator": CalculatorTool()}

_CAPABILITIES = {
    "jsonrpc": "2.0",
    "id": None,
    "result": {
        "protocolVersion": "2024-11-05",
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "reference-agent", "version": "0.1.0"},
    },
}


def _tools_list():
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters,
                }
                for t in _TOOLS.values()
            ]
        },
    }


def _error(code: int, message: str, request_id=None) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _tools_call(params: dict):
    name = params.get("name")
    if not isinstance(name, str) or not name.strip():
        return _error(-32602, "tool name must be a non-empty string")
    arguments = params.get("arguments", {})
    if not isinstance(arguments, dict):
        return _error(-32602, "tool arguments must be an object")
    tool = _TOOLS.get(name)
    if tool is None:
        return _error(-32602, f"unknown tool: {name}")
    try:
        result = tool.run(**arguments)
    except Exception as exc:  # noqa: BLE001
        return _error(-32603, str(exc))
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {"content": [{"type": "text", "text": str(result)}]},
    }


def handle(request: dict) -> dict | None:
    if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not isinstance(request.get("method"), str):
        return _error(-32600, "invalid request")
    rid = request.get("id")
    # bool is a Python int subclass, but JSON-RPC IDs cannot be booleans.
    if type(rid) not in (str, int, float, type(None)) or (type(rid) is float and not math.isfinite(rid)):
        return _error(-32600, "invalid request id")
    if "id" not in request:
        return None  # JSON-RPC notifications never receive a response.
    if not isinstance(request.get("params", {}), dict):
        return _error(-32602, "invalid params", rid)
    method = request.get("method")
    if method == "initialize":
        resp = dict(_CAPABILITIES)
    elif method == "tools/list":
        resp = _tools_list()
    elif method == "tools/call":
        resp = _tools_call(request.get("params", {}))
    else:
        return _error(-32601, f"method not found: {method}", rid)
    resp["id"] = rid
    return resp


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = _error(-32700, "parse error")
        else:
            response = handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
