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
