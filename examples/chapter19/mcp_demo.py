"""Discover and call an MCP 2026-07-28 tool through the official Python SDK."""

import asyncio

from mcp import Client
from mcp.server import MCPServer
from mcp.types import TextContent

server = MCPServer("arithmetic")


@server.tool()
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


async def main() -> None:
    async with Client(server) as client:
        if client.protocol_version != "2026-07-28":
            raise RuntimeError(f"Unexpected MCP version: {client.protocol_version}")
        tools = await client.list_tools()
        print(f"{client.protocol_version}: {', '.join(tool.name for tool in tools.tools)}")

        result = await client.call_tool("add", {"a": 12, "b": 8})
        if result.is_error:
            raise RuntimeError(f"Tool failed: {result.content}")
        for block in result.content:
            if isinstance(block, TextContent):
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
