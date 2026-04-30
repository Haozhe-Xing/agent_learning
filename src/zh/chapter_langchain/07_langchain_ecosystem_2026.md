# 11.7 LangChain 生态 2026

> **本节目标**：了解 LangChain 生态的最新进展，掌握 LangGraph Platform、LangServe、MCP 集成等核心工具，并理解从旧版 AgentExecutor 迁移到 LangGraph 的路径。

---

## LangChain 生态全景

LangChain 已经从一个单一框架发展为一个完整的生态系统。截至 2026 年，LangChain 生态的核心成员如下：

| 工具 | 定位 | 核心价值 |
|------|------|---------|
| **LangChain** | 核心编排框架 | 组件抽象 + LCEL 表达式语言 |
| **LangGraph** | 有状态 Agent 框架 | 图结构编排、循环、人机协作 |
| **LangGraph Platform** | 托管运行服务 | 部署、扩缩容、持久化 |
| **LangServe** | API 部署工具 | 一行代码将 Chain 发布为 REST API |
| **LangSmith** | 可观测性平台 | 追踪、评估、Prompt 管理 |
| **LangChain CLI** | 项目脚手架 | Templates 快速启动模板 |

> 💡 **演进逻辑**：LangChain 负责"定义组件"，LangGraph 负责"编排流程"，LangServe/LangGraph Platform 负责"部署运行"，LangSmith 负责"监控评估"——四层协作覆盖了 Agent 应用的完整生命周期。

---

## LangGraph Platform 托管服务

LangGraph Platform 是 LangGraph 的托管运行环境，解决了 Agent 应用部署中最棘手的问题：**有状态长时运行**。

### 为什么需要 LangGraph Platform？

一个典型的 Agent 应用有以下部署难题：

| 难题 | 传统部署 | LangGraph Platform |
|------|---------|-------------------|
| 长时运行任务 | HTTP 超时、进程崩溃丢失状态 | 内置持久化，自动恢复 |
| 人机协作等待 | 需要自己实现挂起/恢复 | 原生支持 interrupt/resume |
| 并发管理 | 需要自己加锁、限流 | 内置并发控制和队列 |
| 水平扩缩容 | 无状态服务容易扩，有状态难 | State Server 统一管理状态 |
| 流式输出 | 需要 SSE + 反压处理 | 内置流式 API |

### 核心架构

LangGraph Platform 采用三层架构：

```
┌─────────────────────────────────────────┐
│            API Server                    │  ← REST API 入口
│   (部署在 Kubernetes / Cloud Run)        │
├─────────────────────────────────────────┤
│          State Server                    │  ← 状态持久化层
│   (Redis / PostgreSQL / 内存)            │
├─────────────────────────────────────────┤
│         Worker Pool                      │  ← 实际执行 Agent
│   (异步 Worker，可水平扩展)               │
└─────────────────────────────────────────┘
```

### 使用示例

```python
# 1. 定义你的 LangGraph Agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search_docs(query: str) -> str:
    """搜索文档"""
    return f"搜索结果：{query} 的相关文档内容..."

@tool
def run_analysis(data: str) -> str:
    """运行数据分析"""
    return f"分析结果：{data} 的统计摘要..."

tools = [search_docs, run_analysis]
llm = ChatOpenAI(model="gpt-4.1").bind_tools(tools)

def agent_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: MessagesState):
    last = state["messages"][-1]
    if last.tool_calls:
        return "tools"
    return END

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()

# 2. 本地测试
result = app.invoke({"messages": [{"role": "user", "content": "搜索 LangGraph 文档"}]})
print(result["messages"][-1].content)
```

```bash
# 3. 部署到 LangGraph Platform
# 创建 langgraph.json 配置文件
cat > langgraph.json << 'EOF'
{
    "dependencies": ["."],
    "graphs": {
        "agent": "./agent.py:app"
    },
    "env": ".env"
}
EOF

# 部署
langgraph deploy

# 或者使用 Docker 自托管
langgraph build -t my-agent:latest
docker run -p 8000:8000 my-agent:latest
```

```python
# 4. 客户端调用
from langgraph_sdk import get_client

# 连接到 LangGraph Platform
client = get_client(url="http://localhost:8000")

# 创建线程（有状态会话）
thread = await client.threads.create()

# 发送消息
run = await client.runs.create(
    thread_id=thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "帮我分析一下最近的数据"}]},
    stream_mode="values",
)

# 流式接收结果
async for chunk in client.runs.join_stream(
    thread_id=thread["thread_id"],
    run_id=run["run_id"],
    stream_mode="values",
):
    if chunk.data and "messages" in chunk.data:
        last_msg = chunk.data["messages"][-1]
        if isinstance(last_msg, dict) and last_msg.get("content"):
            print(last_msg["content"], end="", flush=True)
```

> 💡 **LangGraph Platform vs 自己部署**：如果你的 Agent 需要人机协作（interrupt/resume）或长时运行，LangGraph Platform 省去了大量基础设施工作。简单场景用 LangServe 即可。

---

## LangServe 部署方案

LangServe 让你用一行代码将 LangChain 应用发布为 REST API——非常适合不需要复杂状态管理的场景。

### 基础部署

```python
# pip install langserve fastapi uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(
    title="LangChain Agent API",
    version="1.0",
)

# 定义 Chain
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。"),
    ("human", "{question}")
])
chain = prompt | llm | StrOutputParser()

# 一行添加 API 路由
add_routes(app, chain, path="/chat")

# 运行：uvicorn server:app --host 0.0.0.0 --port 8000
```

启动后自动获得以下端点：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/chat/invoke` | POST | 同步调用 |
| `/chat/stream` | POST | 流式调用（SSE） |
| `/chat/batch` | POST | 批量调用 |
| `/chat/playground` | GET | 交互式测试页面 |

### 完整的 Agent API 示例

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

app = FastAPI(title="Customer Service Agent API")

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# 工具定义
# ============================

@tool
def search_faq(query: str) -> str:
    """搜索常见问题。"""
    return f"FAQ 结果：{query}"

@tool
def check_order(order_id: str) -> str:
    """查询订单状态。"""
    return f"订单 {order_id}：已发货"

# ============================
# Agent 构建
# ============================

tools = [search_faq, check_order]
llm = ChatOpenAI(model="gpt-4.1", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是客服助手，使用工具帮助用户。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# 会话历史
store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

agent_with_history = RunnableWithMessageHistory(
    agent_executor,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# 添加路由
add_routes(app, agent_with_history, path="/agent")

# 健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

# 启动：uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 客户端调用

```python
# 同步调用
from langserve import RemoteRunnable

remote_chain = RemoteRunnable("http://localhost:8000/chat/")
result = remote_chain.invoke({"role": "助手", "question": "你好"})
print(result)

# 流式调用
for chunk in remote_chain.stream({"role": "助手", "question": "介绍一下自己"}):
    print(chunk, end="", flush=True)
```

```bash
# 使用 curl 测试
curl -X POST http://localhost:8000/chat/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"role": "助手", "question": "你好"}}'
```

---

## LangChain Templates 快速启动模板

LangChain CLI 提供了官方模板，帮你快速创建特定类型的应用：

```bash
# 安装 LangChain CLI
pip install langchain-cli

# 列出可用模板
langchain templates list

# 从模板创建项目
langchain app new my-rag-app --template rag-conversational

# 项目结构
# my-rag-app/
# ├── app/                   # LangServe 服务器
# │   ├── server.py          # FastAPI 入口
# │   └── __init__.py
# ├── chain/                 # 核心逻辑
# │   ├── chain.py           # Chain 定义
# │   └── __init__.py
# ├── pyproject.toml
# └── .env
```

常用模板：

| 模板名 | 用途 | 核心技术 |
|--------|------|---------|
| `rag-conversational` | 对话式 RAG | Retrieval + Memory |
| `extraction-openai-functions` | 信息抽取 | Function Calling |
| `openai-functions-agent` | 通用 Agent | Tools + Agent |
| `pinecone-semantic-search` | 语义搜索 | Pinecone + Embeddings |

---

## LangChain 与 MCP 集成

MCP（Model Context Protocol）是 Anthropic 提出的标准化协议，让 LLM 能以统一的方式连接外部工具和数据源。LangChain 社区已经提供了 MCP 集成。

> 📌 关于 MCP 的详细介绍，请参考第 16 章 [第16章 Agent 通信协议](../chapter_protocol/README.md)。

### 使用 MCP 工具

```python
# pip install langchain-mcp

from langchain_mcp import MCPToolkit
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 连接到 MCP Server
async def create_agent_with_mcp():
    # 方式1：连接到 stdio 模式的 MCP Server
    toolkit = MCPToolkit.from_server_command(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    # 方式2：连接到 SSE 模式的 MCP Server
    # toolkit = MCPToolkit.from_sse_url("http://localhost:3001/sse")

    async with toolkit.session() as session:
        # 自动发现 MCP Server 提供的工具
        mcp_tools = toolkit.get_tools()

        llm = ChatOpenAI(model="gpt-4.1", temperature=0)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你可以使用文件系统工具来帮助用户。"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(llm, mcp_tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=mcp_tools,
            verbose=True,
        )

        result = agent_executor.invoke({
            "input": "读取 /tmp/notes.txt 文件的内容",
            "chat_history": [],
        })
        print(result["output"])

# 运行
import asyncio
asyncio.run(create_agent_with_mcp())
```

### 将 LangChain 工具暴露为 MCP Server

```python
# 将你已有的 LangChain 工具通过 MCP 协议暴露出去
# 这样其他 MCP 兼容的客户端（如 Claude Desktop）也能调用

from langchain_core.tools import tool
from langchain_mcp import create_mcp_server

@tool
def search_internal_docs(query: str) -> str:
    """搜索内部文档库"""
    return f"搜索结果：{query}"

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询"""
    return f"查询结果：[模拟数据]"

# 创建 MCP Server
server = create_mcp_server(
    tools=[search_internal_docs, query_database],
    server_name="internal-tools",
    server_version="1.0.0",
)

# 启动 Server
# server.run(transport="stdio")   # Claude Desktop 集成
# server.run(transport="sse", port=3001)  # SSE 模式
```

> ⚠️ **MCP 集成的价值**：MCP 让 LangChain 工具不再局限于 LangChain 生态内部——Claude Desktop、其他 MCP 兼容客户端都可以调用你的工具。这对于企业内部工具共享特别有价值。

---

## LangChain 2025-2026 重大变更

### 从 LangChain v0.1 到 v0.3 的架构演变

LangChain 在 2024-2025 年间经历了剧烈的架构变化。如果你在维护旧代码，以下迁移指南至关重要：

| 变更项 | v0.1（旧） | v0.3（新） | 影响 |
|--------|-----------|-----------|------|
| **包结构** | `from langchain import ...` | `from langchain_openai import ...` | 所有导入路径 |
| **链构建** | `LLMChain(llm=..., prompt=...)` | `prompt \| llm \| parser` | 核心范式 |
| **Agent** | `AgentExecutor` | `LangGraph` | Agent 编排 |
| **输出解析** | `output_key` 参数 | LCEL 自动传递 | 链输出 |
| **回调** | `callbacks` 参数 | `config={"callbacks": [...]}` | 回调机制 |
| **消息类型** | `HumanMessage(content=...)` | 同上，但新增 `.type` 属性 | 消息处理 |

### 关键废弃 API 速查

```python
# ❌ 已废弃
from langchain.llms import OpenAI                    # → 使用 ChatOpenAI
from langchain.chains import LLMChain                # → 使用 LCEL (prompt | llm)
from langchain.chains import RetrievalQA             # → 使用 LCEL + retriever
from langchain.agents import initialize_agent        # → 使用 create_openai_tools_agent
from langchain.chat_models import ChatOpenAI         # → 使用 langchain_openai.ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings    # → 使用 langchain_openai.OpenAIEmbeddings
from langchain.vectorstores import Chroma            # → 使用 langchain_chroma.Chroma

# ✅ 推荐写法
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma

chain = prompt | llm | StrOutputParser()  # LCEL
```

---

## 迁移指南：从 AgentExecutor 迁移到 LangGraph

这是最重要的迁移——LangChain 官方推荐新项目使用 LangGraph 构建 Agent。

### 为什么迁移？

| AgentExecutor | LangGraph |
|---------------|-----------|
| 固定的 `observe → act → observe` 循环 | 自由定义任意拓扑的图 |
| 难以实现"先审批再执行"等流程 | 原生支持 interrupt/resume |
| 循环控制只有 `max_iterations` | 条件路由、循环、分支完整支持 |
| 状态管理受限 | 完全自定义 State |
| 无法表达并行步骤 | 原生并行节点 |

### 迁移对照表

```python
# ========================================
# 旧版：AgentExecutor
# ========================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"结果：{query}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return f"结果：{eval(expression)}"

tools = [search, calculate]
llm = ChatOpenAI(model="gpt-4.1", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手，使用工具帮助用户。"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_openai_tools_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    handle_parsing_errors=True,
)

result = agent_executor.invoke({
    "input": "搜索 Python 最新版本",
    "chat_history": [],
})
```

```python
# ========================================
# 新版：LangGraph
# ========================================

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"结果：{query}"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式"""
    return f"结果：{eval(expression)}"

tools = [search, calculate]
llm = ChatOpenAI(model="gpt-4.1", temperature=0).bind_tools(tools)

# 定义 Agent 节点
def agent_node(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 构建图
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)  # 自动判断是否调用工具
graph.add_edge("tools", "agent")  # 工具执行后回到 Agent

app = graph.compile()

# 调用
result = app.invoke({
    "messages": [{"role": "user", "content": "搜索 Python 最新版本"}]
})
print(result["messages"][-1].content)
```

### 关键迁移点

| AgentExecutor 概念 | LangGraph 对应 | 说明 |
|-------------------|---------------|------|
| `AgentExecutor(...)` | `graph.compile()` | 编译图 |
| `max_iterations` | 图的循环结构本身 | 不再需要手动限制 |
| `handle_parsing_errors` | 工具节点的错误处理 | 更细粒度 |
| `return_intermediate_steps` | State 中自动保留 | 消息即状态 |
| `verbose=True` | LangSmith 追踪 | 更好的调试方式 |
| 会话历史 | `MemorySaver` / `checkpointer` | 持久化方案 |
| `agent_scratchpad` | `MessagesState` 自动管理 | 无需手动 |

> 💡 **迁移建议**：
> - 新项目直接用 LangGraph，不要再用 AgentExecutor
> - 旧项目可以分步迁移——先替换 Agent 核心循环，工具定义不需要改
> - 详细的 LangGraph 教程请参考第 12 章 [第12章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md)

---

## 小结

LangChain 生态在 2025-2026 年的核心演进方向是**从"框架"走向"平台"**：

| 演进方向 | 具体表现 |
|---------|---------|
| **编排进化** | AgentExecutor → LangGraph 图编排 |
| **部署简化** | LangServe 一行部署 → LangGraph Platform 托管运行 |
| **协议开放** | MCP 集成，工具不再局限于 LangChain 生态 |
| **可观测性** | LangSmith 从"追踪"扩展为"评估+管理"平台 |
| **架构稳定** | v0.3 移除废弃 API，LCEL 成为标准范式 |

> 💡 **与本书其他章节的关系**：
> - 第 12 章 [第12章 LangGraph：构建有状态的 Agent](../chapter_langgraph/README.md) 深入讲解 LangGraph 图编排
> - 第 16 章 [第16章 Agent 通信协议](../chapter_protocol/README.md) 详解 MCP 协议
> - 第 19 章 [第19章 部署与生产化](../chapter_deployment/README.md) 讨论更完整的部署方案

---

*下一节：[11.8 LangChain 生产化模式](./08_production_patterns.md)*

---

## 参考文献

[1] LangChain Team. LangGraph Platform Documentation. https://langchain-ai.github.io/langgraph/cloud, 2025.

[2] LangChain Team. LangServe Documentation. https://python.langchain.com/docs/langserve, 2025.

[3] LangChain Team. LangChain MCP Adapters. https://github.com/langchain-ai/langchain-mcp-adapters, 2025.

[4] LangChain Team. Migration Guide: AgentExecutor to LangGraph. https://python.langchain.com/docs/migration, 2025.
