"""A minimal, runnable MCP (Model Context Protocol) server over stdio.

This is a self-contained JSON-RPC 2.0 server. It fixes the old book bug where
the example returned tools bound to an already-closed session: here the
server owns the tools and stays alive for the whole stdio session.

Run with:  python -m reference_agent.mcp_server
"""

from __future__ import annotations

import json
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


def _tools_call(params: dict):
    name = params.get("name")
    tool = _TOOLS.get(name)
    if tool is None:
        return {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32602, "message": f"unknown tool: {name}"},
        }
    try:
        result = tool.run(**params.get("arguments", {}))
    except Exception as exc:  # noqa: BLE001
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32603, "message": str(exc)}}
    return {
        "jsonrpc": "2.0",
        "id": None,
        "result": {"content": [{"type": "text", "text": str(result)}]},
    }


def handle(request: dict) -> dict:
    method = request.get("method")
    rid = request.get("id")
    if method == "initialize":
        resp = dict(_CAPABILITIES)
    elif method == "tools/list":
        resp = _tools_list()
    elif method == "tools/call":
        resp = _tools_call(request.get("params", {}))
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"method not found: {method}"}}
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
            continue
        response = handle(request)
        if response is not None:
            sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
