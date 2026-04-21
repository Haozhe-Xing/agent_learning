# 2.2 关键库安装

本节介绍 Agent 开发生态中最重要的库，并说明它们的用途和安装方式。

## Agent 开发核心库全景

![Agent开发技术栈全景](../svg/chapter_setup_02_tech_stack.svg)

## 一键安装：Agent 开发标准套件

```bash
# 创建并激活虚拟环境
uv venv && source .venv/bin/activate

# 核心依赖（必装）
uv add openai langchain langchain-openai langchain-community python-dotenv pydantic

# LangGraph（有状态 Agent）
uv add langgraph

# OpenAI Agents SDK（轻量级多 Agent 框架）
uv add openai-agents

# 向量数据库（本地开发用 chromadb 就够）
uv add chromadb

# 工具库
uv add requests beautifulsoup4 wikipedia-api

# MCP 协议支持
uv add mcp

# 工程库
uv add tenacity rich
```

或者使用 pip：

```bash
pip install openai langchain langchain-openai langchain-community langgraph \
            openai-agents chromadb mcp python-dotenv pydantic tenacity rich
```

## 各库详解

### OpenAI SDK

```python
# 安装：pip install openai
# 用途：调用 GPT 系列模型

from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# 验证安装
import openai
print(f"OpenAI SDK 版本：{openai.__version__}")  # 应该 >= 1.0.0
```

### LangChain

```python
# 安装：pip install langchain langchain-openai
# 用途：Agent 框架、链式调用、工具集成

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 基础模型调用
llm = ChatOpenAI(model="gpt-4.1-mini")
response = llm.invoke([HumanMessage(content="你好！")])
print(response.content)

# 验证安装
import langchain
print(f"LangChain 版本：{langchain.__version__}")  # 应该 >= 0.3.0
```

### LangGraph

```python
# 安装：pip install langgraph
# 用途：构建有状态的 Agent 工作流

from langgraph.graph import StateGraph, END

# 后续章节详细讲解，先验证能导入
import langgraph
print(f"LangGraph 版本：{langgraph.__version__}")
```

### ChromaDB（向量数据库）

```python
# 安装：pip install chromadb
# 用途：本地向量存储，用于 RAG 场景

import chromadb

# 创建本地向量数据库
client = chromadb.Client()
collection = client.create_collection("test")

# 添加文档
collection.add(
    documents=["Python 是一种解释型编程语言", "LangChain 是 Agent 框架"],
    ids=["doc1", "doc2"]
)

# 查询相似文档
results = collection.query(
    query_texts=["如何学习 Python？"],
    n_results=1
)
print(results)

import chromadb
print(f"ChromaDB 版本：{chromadb.__version__}")
```

### Pydantic（数据验证）

```python
# 安装：pip install pydantic
# 用途：数据模型定义、输入验证、Agent 输出解析

from pydantic import BaseModel, Field
from typing import Optional, List

class TaskInfo(BaseModel):
    """任务信息模型"""
    title: str = Field(..., description="任务标题")
    priority: str = Field(default="medium", pattern="^(high|medium|low)$")
    tags: List[str] = Field(default_factory=list)
    deadline: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "完成项目报告",
                "priority": "high",
                "tags": ["工作", "报告"],
                "deadline": "2024-12-31"
            }
        }

# 验证数据
task = TaskInfo(title="写代码", priority="high", tags=["开发"])
print(task.model_dump())
print(task.model_dump_json())

import pydantic
print(f"Pydantic 版本：{pydantic.__version__}")  # 应该 >= 2.0
```

### Rich（美化终端输出）

```python
# 安装：pip install rich
# 用途：漂亮的终端输出，调试 Agent 时非常有用

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

# 打印带颜色的信息
console.print("[bold green]✅ Agent 启动成功[/bold green]")
console.print("[yellow]⚠️ 注意：Token 消耗较多[/yellow]")
console.print("[red]❌ 工具调用失败[/red]")

# 展示代码
code = """
def hello_agent():
    return "Hello, World!"
"""
syntax = Syntax(code, "python", theme="monokai")
console.print(Panel(syntax, title="Agent 代码"))

# 展示表格
table = Table(title="工具列表")
table.add_column("工具名", style="cyan")
table.add_column("描述", style="green")
table.add_column("状态", style="yellow")

table.add_row("search", "搜索互联网", "✅ 可用")
table.add_row("calculator", "数学计算", "✅ 可用")
table.add_row("email", "发送邮件", "❌ 未配置")

console.print(table)
```

### Tenacity（重试机制）

```python
# 安装：pip install tenacity
# 用途：API 调用的重试逻辑，生产环境必备

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from openai import RateLimitError, APIError

@retry(
    stop=stop_after_attempt(3),           # 最多重试3次
    wait=wait_exponential(min=1, max=10), # 指数退避：1s, 2s, 4s...
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True
)
def robust_api_call(messages: list) -> str:
    """带重试的 API 调用"""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages
    )
    return response.choices[0].message.content

# 自动处理速率限制和临时错误
result = robust_api_call([{"role": "user", "content": "你好"}])
```

## 完整安装验证脚本

创建以下脚本来一次性验证所有依赖：

```python
# verify_installation.py
"""运行此脚本验证所有依赖是否正确安装"""

import sys
from rich.console import Console
from rich.table import Table

console = Console()

def check_package(package_name: str, import_name: str = None) -> tuple:
    """检查包是否安装"""
    import_name = import_name or package_name.replace("-", "_")
    try:
        import importlib
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

# 检查列表
packages = [
    ("openai", "openai"),
    ("langchain", "langchain"),
    ("langchain-openai", "langchain_openai"),
    ("langgraph", "langgraph"),
    ("chromadb", "chromadb"),
    ("pydantic", "pydantic"),
    ("python-dotenv", "dotenv"),
    ("tenacity", "tenacity"),
    ("rich", "rich"),
    ("requests", "requests"),
    ("beautifulsoup4", "bs4"),
]

table = Table(title="依赖安装检查", show_header=True)
table.add_column("包名", style="cyan")
table.add_column("版本", style="yellow")
table.add_column("状态", style="green")

all_ok = True
for pkg_name, import_name in packages:
    installed, version = check_package(pkg_name, import_name)
    status = "✅ 已安装" if installed else "❌ 未安装"
    version_str = version if installed else "-"
    table.add_row(pkg_name, version_str, status)
    if not installed:
        all_ok = False

console.print(table)

if all_ok:
    console.print("\n[bold green]🎉 所有依赖已正确安装！可以开始 Agent 开发了。[/bold green]")
else:
    console.print("\n[bold red]⚠️ 部分依赖未安装，请运行：pip install <缺失的包>[/bold red]")

# 检查 Python 版本
python_version = sys.version_info
if python_version >= (3, 10):
    console.print(f"[green]✅ Python {python_version.major}.{python_version.minor} - 版本满足要求[/green]")
else:
    console.print(f"[red]❌ Python {python_version.major}.{python_version.minor} - 需要 >= 3.10[/red]")
```

运行验证：

```bash
python verify_installation.py
```

---

## 小结

Agent 开发的核心依赖：

| 分类 | 推荐库 | 用途 |
|------|-------|------|
| LLM 接口 | `openai` | 调用 GPT 模型 |
| Agent 框架 | `langchain` + `langgraph` | 构建 Agent 逻辑 |
| 向量存储 | `chromadb` | RAG 文档检索 |
| 数据验证 | `pydantic` | 结构化输出 |
| 工程工具 | `tenacity` + `rich` | 重试 + 漂亮日志 |

---

*下一节：[2.3 API Key 管理与安全最佳实践](./03_api_key_management.md)*
