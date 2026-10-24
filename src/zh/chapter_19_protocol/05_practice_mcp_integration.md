# 19.5 实战：基于 MCP 的工具集成

本节讲清楚一件事：**怎样让 Agent 真正用上一个 MCP 工具**，而不是只画出架构图。

![MCP 工具集成架构](../svg/chapter_19_protocol_05_mcp_integration.svg)

MCP 的核心价值在于**标准化和解耦**：工具的实现（MCP Server）和工具的使用（MCP Client / Agent）完全分离。你可以用任何语言写工具服务器，任何支持 MCP 的 Agent 框架都能直接用，不必为每个框架单独适配。

> ⚠️ **一个真实踩过的坑**：MCP 工具通常持有对 `ClientSession` 的引用。如果像下面这样把 `async with ClientSession(...) as session:` 写在 `load_mcp_tools()` 内部、并在 `with` 块里 `return tools`，那么函数一返回，上下文退出、session 关闭，返回的工具就绑死了一个已失效的 session——后面调用必然失败。**正确做法是让 session 的生命周期覆盖整个 Agent 运行。**

---

## 一、先写一个能跑的 MCP Server

仓库 `reference-agent/src/reference_agent/mcp_server.py` 是一个**最小、可运行、已测试**的 MCP Server（stdio + JSON-RPC 2.0）。它自己持有工具，生命周期与进程一致，不存在"返回已关闭 session"的问题。可直接用：

```bash
# 在项目根目录 reference-agent/ 下
python -m reference_agent.mcp_server
# 然后向 stdin 逐行发送 JSON-RPC：initialize / tools/list / tools/call
```

其核心结构（节选，完整见源码）：

```python
# reference-agent/src/reference_agent/mcp_server.py
_TOOLS = {"calculator": CalculatorTool()}

def handle(request: dict) -> dict:
    method = request.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": request.get("id"),
                "result": {"protocolVersion": "2024-11-05",
                           "capabilities": {"tools": {}},
                           "serverInfo": {"name": "reference-agent", "version": "0.1.0"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request.get("id"),
                "result": {"tools": [{"name": t.name, "description": t.description,
                                      "inputSchema": t.parameters} for t in _TOOLS.values()]}}
    if method == "tools/call":
        name = request["params"]["name"]
        tool = _TOOLS.get(name)
        if tool is None:
            return error(f"unknown tool: {name}")
        result = tool.run(**request["params"].get("arguments", {}))
        return {"jsonrpc": "2.0", "id": request.get("id"),
                "result": {"content": [{"type": "text", "text": str(result)}]}}
```

---

## 二、正确的 Client / Agent 集成方式

关键原则：**session 的存活范围要覆盖 Agent 的整个执行**。下面用官方 `mcp` SDK 给出正确形态（伪代码骨架，真实运行需安装 `mcp` 与对应 LLM SDK）：

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_agent_with_mcp(server_command, server_args, user_input):
    server_params = StdioServerParameters(command=server_command, args=server_args)

    # ✅ session 与 agent 执行在同一个 async with 内，生命周期一致
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_tools = await session.list_tools()

            # 在这里构建并运行 Agent，session 全程存活
            answer = await agent_run(user_input, mcp_tools, session)
            return answer

# 注意：不要像旧版本那样在 async with 内部 return tools 后退出上下文，
# 否则 tools 持有的 session 已经关闭，调用会失败。
```

如果你用的是 LangChain / LangGraph，思路相同：先把 `ClientSession` 打开并保活，再把 `session.call_tool` 包成工具注入 Agent，**Agent 运行期间不要关闭 session**。第 13 章的 LangGraph 实战与第 18 章的多 Agent 系统都可沿用这个约束。

---

## 三、工具安全：LLM 生成的参数不可信

MCP 工具常涉及文件、数据库、HTTP，而参数由 LLM 生成，可能被 Prompt 注入利用。下面是一段**示意性**的只读 SQL 工具校验（仅展示思路，生产环境请结合具体数据库驱动与参数化查询）：

```python
def safe_query(db_path: str, sql: str) -> str:
    sql = sql.strip()
    sql_upper = sql.upper()
    # 防线 1：只允许 SELECT 开头
    if not sql_upper.startswith("SELECT"):
        raise PermissionError("只允许 SELECT 查询")
    # 防线 2：拦截危险关键词
    for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"):
        if kw in sql_upper:
            raise PermissionError(f"SQL 含禁止关键词: {kw}")
    # 防线 3：禁止多语句（去掉结尾分号后不应再有分号）
    if ";" in sql.rstrip(";"):
        raise PermissionError("不允许执行多条 SQL 语句")
    # 防线 4：限制返回行数，防数据爆量
    import sqlite3
    conn = sqlite3.connect(db_path)
    rows = conn.cursor().execute(sql).fetchmany(100)
    conn.close()
    return str([dict(r) for r in rows])
```

四道防线分别挡住：以非 SELECT 开头的破坏性语句、夹带写关键词的混合攻击、分号拼接的第二句恶意 SQL、以及 `SELECT *` 拉爆全表。

---

## 小结

- MCP = **Agent ↔ 工具/数据源** 的连接标准（"USB 接口"），价值在标准化与解耦。
- 集成时的硬规则：**session 生命周期必须覆盖 Agent 运行**，绝不在 `async with` 内部返回工具后退出上下文。
- 真实可运行的最小 Server 见 `reference-agent/src/reference_agent/mcp_server.py`（已带测试）。
- 工具参数来自 LLM，必须按"不可信输入"做校验与最小化权限。

> 📌 关于 MCP / A2A / ANP 的"安装量""市场份额"等说法，请以官方文档与发布方最新数据为准，本书不引用无法独立核验的数字。

---

## 📝 本章练习

读完本章，先合上书用自己的话回答下面的问题，再展开参考答案对照。

**练习 1（概念）**：本章把 MCP 比作"AI 世界的 USB-C 接口"。请解释：在 MCP 出现之前，给 Agent 接工具有什么痛点？MCP 的"标准化 + 解耦"具体解决了什么问题？

<details>
<summary>参考答案</summary>

**MCP 之前的痛点：工具接口各家不兼容。** 同一个 search 工具，OpenAI Function Calling、LangChain、Anthropic 的格式各不相同，每接入一个新框架就得重写一遍。

**MCP 解决的核心问题：标准化 + 解耦。** 它规定了统一的工具描述格式（name / description / inputSchema）和调用协议；工具的**实现**（MCP Server）与**使用**（MCP Client / Agent）被彻底分开，互不关心对方用什么语言、什么框架。

**为什么"写一次，到处用"？** 接口标准化后，一个 Python 写的 MCP 天气服务器，任何支持 MCP 的客户端都能直接连上来用，无需单独适配。这正是 MCP 能成为主流工具协议的根本原因。

</details>

**练习 2（辨析）**：有同学说"MCP、A2A、ANP 都是 Agent 通信协议，功能重叠，最后肯定只有一个能活下来。" 请反驳：用一句类比概括三者定位，说明为什么是"互补而非竞争"，并描述一个三者协同的场景。

<details>
<summary>参考答案</summary>

**这个观点是错的——三者处在不同抽象层，是互补关系。**

| 协议 | 一句话类比 | 解决的核心问题 |
|------|-----------|--------------|
| **MCP** | Agent 的"USB 接口" | Agent ↔ **工具/数据源** 怎么连接 |
| **A2A** | Agent 的"即时通讯" | Agent ↔ **Agent** 怎么协作（任务委派、流式） |
| **ANP** | Agent 的"开放网络" | **大规模 Agent 网络**怎么发现、验证、安全通信 |

**协同场景**：用户向公司 A 的客服 Agent 咨询退货——ANP 负责在开放网络中发现并验证公司 B 的物流 Agent、公司 C 的支付 Agent；A2A 负责把"查物流""办退款"等任务委派给 B、C 并追踪状态；MCP 负责物流/支付 Agent 各自调用自己的内部工具。三层叠在一起，构成完整生态。

</details>

**练习 3（动手）**：实现一个只读 SQL 工具 `query_database(db_path, sql)`，只放行 `SELECT`、拦截危险关键词、禁止多语句、限制返回行数。写出核心校验逻辑，并解释每道防线分别挡住什么攻击。

<details>
<summary>参考答案</summary>

核心思路见上文"三、工具安全"代码：只放行 SELECT、拦截写关键词、禁止多语句、限制 100 行。四道防线分别挡住破坏性语句、夹带写操作的混合攻击、分号拼接的第二句恶意 SQL，以及 `SELECT *` 的数据爆量。

</details>

---

*下一章：[第20章 Agent 的评估与优化](../chapter_20_evaluation/README.md)*
