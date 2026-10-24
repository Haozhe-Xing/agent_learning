import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reference_agent.mcp_server import handle


def test_initialize():
    resp = handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert resp["result"]["serverInfo"]["name"] == "reference-agent"


def test_tools_list():
    resp = handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in resp["result"]["tools"]]
    assert "calculator" in names


def test_tools_call():
    resp = handle(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "calculator", "arguments": {"expression": "12*8"}},
        }
    )
    assert resp["result"]["content"][0]["text"] == "96"


def test_unknown_method():
    resp = handle({"jsonrpc": "2.0", "id": 4, "method": "bogus"})
    assert "error" in resp


def test_notification_has_no_response():
    assert handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_invalid_request_does_not_crash_server():
    assert handle([])["error"]["code"] == -32600
    assert handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []})["error"]["code"] == -32602


@pytest.mark.parametrize("name", [[], {}, None, 1, True, "", "  "])
def test_tools_call_rejects_invalid_name(name):
    response = handle({"jsonrpc": "2.0", "id": "bad-name", "method": "tools/call",
                       "params": {"name": name, "arguments": {"expression": "12*8"}}})
    assert response["id"] == "bad-name"
    assert response["error"]["code"] == -32602


@pytest.mark.parametrize("arguments", [[], None, "12*8", 96, True])
def test_tools_call_rejects_non_object_arguments(arguments):
    response = handle({"jsonrpc": "2.0", "id": "bad-arguments", "method": "tools/call",
                       "params": {"name": "calculator", "arguments": arguments}})
    assert response["id"] == "bad-arguments"
    assert response["error"]["code"] == -32602


@pytest.mark.parametrize("request_id", [[], {}, True, False, float("nan"), float("inf")])
def test_invalid_request_id_is_not_echoed(request_id):
    response = handle({"jsonrpc": "2.0", "id": request_id, "method": "tools/list"})
    assert response["id"] is None
    assert response["error"]["code"] == -32600


@pytest.mark.parametrize("request_id", [None, 0, 3.5, "request-1"])
def test_valid_request_id_is_preserved(request_id):
    response = handle({"jsonrpc": "2.0", "id": request_id, "method": "tools/list"})
    assert response["id"] == request_id
    assert "result" in response


def test_unknown_tool_returns_invalid_params():
    response = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "missing", "arguments": {}}})
    assert response["error"]["code"] == -32602


def test_stdio_recovers_from_malformed_requests_in_same_process():
    requests = [
        "{broken json",
        json.dumps([]),
        json.dumps({"jsonrpc": "2.0", "id": {}, "method": "tools/list"}),
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": None}),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": []}}),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": {}}}),
        json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                    "params": {"name": "calculator", "arguments": []}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json.dumps({"jsonrpc": "2.0", "id": "after-errors", "method": "tools/call",
                    "params": {"name": "calculator", "arguments": {"expression": "12*8"}}}),
    ]
    source = str(Path(__file__).resolve().parents[1] / "src")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")])))
    process = subprocess.run([sys.executable, "-m", "reference_agent.mcp_server"],
                             input="\n".join(requests) + "\n", capture_output=True, text=True,
                             timeout=5, check=True, env=env)
    responses = [json.loads(line) for line in process.stdout.splitlines()]
    assert process.stderr == ""
    assert [response["error"]["code"] for response in responses[:-1]] == [
        -32700, -32600, -32600, -32602, -32602, -32602, -32602,
    ]
    assert responses[-1] == {"jsonrpc": "2.0", "id": "after-errors",
                             "result": {"content": [{"type": "text", "text": "96"}]}}
