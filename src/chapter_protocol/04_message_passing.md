# Agent 间的消息传递与状态共享

> **本节目标**：学习 Agent 间消息传递的实现方式，并通过 MCP 工具集成实战加深理解。

![Agent 间消息传递与状态共享架构](../svg/chapter_protocol_04_message_passing.svg)

---

## 生产级 MCP 工具服务器

```python
# production_mcp_server.py
"""
完整的生产级 MCP 工具服务器
包含：文件操作、数据库查询、HTTP请求
"""

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, ListToolsResult
import json
import os
import sqlite3
import requests
import asyncio

server = Server("production-tools-server")

# ============================
# 工具定义
# ============================

TOOLS = [
    Tool(
        name="read_file",
        description="读取本地文件内容。支持 .txt .md .py .json .csv 格式。",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件相对路径"}
            },
            "required": ["path"]
        }
    ),
    Tool(
        name="write_file",
        description="写入内容到文件（覆盖写入）。",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "文件内容"}
            },
            "required": ["path", "content"]
        }
    ),
    Tool(
        name="query_database",
        description="查询 SQLite 数据库（只允许 SELECT 语句）。",
        inputSchema={
            "type": "object",
            "properties": {
                "db_path": {"type": "string", "description": "数据库文件路径"},
                "sql": {"type": "string", "description": "SELECT SQL 语句"}
            },
            "required": ["db_path", "sql"]
        }
    ),
    Tool(
        name="http_get",
        description="发送 HTTP GET 请求获取数据。",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "请求 URL"},
                "headers": {"type": "object", "description": "请求头（可选）"}
            },
            "required": ["url"]
        }
    )
]

@server.list_tools()
async def list_tools() -> ListToolsResult:
    return ListToolsResult(tools=TOOLS)

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    try:
        if name == "read_file":
            path = arguments["path"]
            # 安全检查：防止路径遍历
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(os.getcwd()):
                raise PermissionError("不允许访问当前目录外的文件")
            
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return CallToolResult(
                content=[TextContent(type="text", text=content[:10000])]
            )
        
        elif name == "write_file":
            path = arguments["path"]
            content = arguments["content"]
            
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return CallToolResult(
                content=[TextContent(type="text", text=f"已写入 {len(content)} 字符到 {path}")]
            )
        
        elif name == "query_database":
            db_path = arguments["db_path"]
            sql = arguments["sql"].strip()
            
            # 安全检查：只允许 SELECT，禁止危险关键词
            sql_upper = sql.upper()
            if not sql_upper.startswith("SELECT"):
                raise PermissionError("只允许 SELECT 查询")
            
            # 检查是否包含危险操作（防止 SELECT 后跟分号执行其他语句）
            dangerous_keywords = [
                "DROP", "DELETE", "UPDATE", "INSERT", 
                "ALTER", "CREATE", "TRUNCATE", "EXEC",
            ]
            for keyword in dangerous_keywords:
                if keyword in sql_upper:
                    raise PermissionError(f"SQL 中包含禁止的关键词: {keyword}")
            
            # 禁止多语句执行（分号分隔）
            if ";" in sql.rstrip(";"):
                raise PermissionError("不允许执行多条 SQL 语句")
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchmany(100)  # 最多100行
            conn.close()
            
            result = [dict(row) for row in rows]
            return CallToolResult(
                content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
            )
        
        elif name == "http_get":
            url = arguments["url"]
            headers = arguments.get("headers", {})
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            content = response.text[:5000]  # 限制返回长度
            return CallToolResult(
                content=[TextContent(type="text", text=content)]
            )
        
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"未知工具：{name}")],
                isError=True
            )
    
    except Exception as e:
        return CallToolResult(
            content=[TextContent(type="text", text=f"工具执行失败：{str(e)}")],
            isError=True
        )

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
```

## 在 Claude Desktop 中使用 MCP

MCP Server 编写完成后，需要在客户端（Host）中注册才能使用。以 Claude Desktop 为例，只需在配置文件中指定 MCP Server 的启动命令和路径。Claude 会在启动时自动连接这些 Server，并在对话中展示可用的工具列表。

```json
// ~/.config/claude/claude_desktop_config.json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["/path/to/production_mcp_server.py"],
      "env": {
        "PYTHONPATH": "/path/to/your/project"
      }
    }
  }
}
```

---

## 本章小结

Agent 通信协议的核心：

| 协议 | 定位 | 主要场景 |
|------|------|---------|
| MCP | LLM ↔ 工具/数据源 | 标准化工具调用 |
| A2A | Agent ↔ Agent | 跨框架 Agent 协作 |

两者互补：MCP 解决工具集成，A2A 解决 Agent 互操作。

---

*下一节：[15.5 实战：基于 MCP 的工具集成](./05_practice_mcp_integration.md)*
