# 11.1 LangChain 架构全景

LangChain 是一个模块化的 LLM 应用开发框架，核心设计思想是**通过标准化接口组合各类组件**，让开发者专注于业务逻辑。

## 核心组件体系

![LangChain核心组件体系](../svg/chapter_langchain_01_langchain_components.svg)

## 架构演进：从单体到分包

LangChain 自 2022 年底发布以来经历了三次重大架构变革，理解这些变化有助于你读懂网上不同时期的教程，避免踩「API 已废弃」的坑。

| 阶段 | 版本 | 核心变化 | 关键特征 |
|------|------|---------|---------|
| **v0.0.x（2022.11—2023.12）** | 单体包 | 所有功能在一个 `langchain` 包中 | `from langchain.llms import OpenAI` |
| **v0.1.x（2024.01—2024.06）** | 分包架构 | 拆分出 `langchain-core`、`langchain-community` | 双重导入路径并存，旧 API 标记为 deprecated |
| **v0.2.x（2024.07—2024.12）** | LCEL 优先 | LCEL 成为标准范式，`LLMChain` 等老式链被移除 | Pydantic V2 支持，Python 3.8 停止支持 |
| **v0.3.x（2025.01—至今）** | 稳定期 | 完全移除废弃 API，集成包独立发布 | `langchain-openai`、`langchain-anthropic` 等独立版本管理 |

### 分包设计理念

LangChain 0.3 的包结构遵循「分层依赖」原则 [1]：

| 层级 | 包名 | 职责 |
|------|------|------|
| 核心层（最底层，最稳定） | `langchain-core` | Runnable 协议、消息类型、Prompt 模板，几乎不做 breaking change |
| 编排层（依赖 core） | `langchain` | Chain 组合、Agent 逻辑、回调系统，提供高层抽象 |
| 集成层（各自独立发版） | `langchain-openai`、`langchain-anthropic`、`langchain-community` | 各 LLM / 工具提供商的具体实现 |

**设计初衷**：

- **`langchain-core`**：只定义接口和协议（如 `Runnable`、`BaseChatModel`、`BaseTool`），保证接口稳定性。当你编写自定义组件时，只需依赖 core 即可。
- **`langchain`**：提供编排能力——如何把多个 Runnable 组合成 Chain、如何构建 Agent。这一层负责「胶水逻辑」。
- **集成包**：每个 LLM 提供商一个包（`langchain-openai`、`langchain-anthropic`、`langchain-google-genai`），各自独立发版，不会因为某个提供商的 API 变更影响其他用户。

> 💡 **实践建议**：新项目一律使用 `from langchain_openai import ChatOpenAI` 这样的分包导入，不要使用 `from langchain.chat_models import ChatOpenAI`（这是旧路径的兼容别名，已标记废弃）。

### Runnable 协议

Runnable 是 LangChain 0.2+ 引入的**最核心抽象**——所有组件（LLM、Prompt、Parser、Tool、Retriever）都实现了统一的 Runnable 接口 [2]。这意味着它们共享一套完全一致的调用方式：

```python
from langchain_core.runnables import Runnable

# 所有 Runnable 都支持以下方法：
# runnable.invoke(input)         ← 同步调用，返回单个结果
# runnable.ainvoke(input)        ← 异步调用
# runnable.stream(input)         ← 流式输出，返回生成器
# runnable.astream(input)        ← 异步流式
# runnable.batch([input1, ...])  ← 批量调用
# runnable.abatch([input1, ...]) ← 异步批量

# Runnable 组合：用 | 管道符连接（即 LCEL）
chain = prompt | llm | parser
# 等价于 chain = RunnableSequence(prompt, llm, parser)

# 并行执行：RunnableParallel
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    translation=translate_chain,
)
# 同一个输入同时执行两条链，返回 {"summary": ..., "translation": ...}
```

**为什么 Runnable 协议如此重要？**

1. **可组合性**：任何 Runnable 都可以用 `|` 串联或用 `RunnableParallel` 并联，构建任意复杂的处理管道
2. **流式默认**：`.stream()` 方法让所有组件都支持流式输出——对于需要实时返回的 Agent 应用至关重要
3. **可观测性**：内置回调系统（`callbacks`）可以追踪每个 Runnable 的输入输出，配合 LangSmith 实现全链路监控
4. **类型安全**：每个 Runnable 有明确的 `input_schema` 和 `output_schema`（基于 Pydantic），支持编译期类型检查

### LangChain 与同类框架对比

在选择开发框架时，了解不同框架的定位有助于做出合理选择：

| 框架 | 核心定位 | 最适合场景 | 社区活跃度 |
|------|---------|-----------|-----------|
| **LangChain** | 通用 LLM 编排 | 需要大量集成的企业应用 | ⭐⭐⭐⭐⭐ |
| **LlamaIndex** | 数据连接 + RAG | 文档问答、知识库 | ⭐⭐⭐⭐ |
| **Haystack** | 搜索 + RAG Pipeline | 搜索增强型应用 | ⭐⭐⭐ |
| **Semantic Kernel** | 微软生态集成 | Azure + C# 项目 | ⭐⭐⭐ |
| **原生 API** | 无框架依赖 | 简单原型、极致性能 | — |

> 📌 **选择建议**：如果你需要快速接入多种 LLM 和工具，LangChain 的集成生态是最大优势；如果你的场景以 RAG 为核心，LlamaIndex 的文档处理能力更专精；如果追求极致可控性，直接使用原生 API。在本书的实战项目中，我们选择 LangChain + LangGraph 组合，因为它在编排灵活性和社区支持上表现最均衡。

---

## 快速上手

```python
# pip install langchain langchain-openai langchain-community

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ============================
# 1. 基础模型调用
# ============================

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)

# 直接调用
response = llm.invoke([HumanMessage(content="你好！")])
print(response.content)

# ============================
# 2. 提示词模板
# ============================

# ChatPromptTemplate：推荐方式
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专注于{domain}领域。"),
    ("user", "{question}")
])

# 格式化
formatted = prompt.format_messages(
    role="Python 专家",
    domain="机器学习",
    question="如何用 sklearn 训练一个分类器？"
)

response = llm.invoke(formatted)
print(response.content)

# ============================
# 3. 输出解析器
# ============================

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class ProductInfo(BaseModel):
    name: str = Field(description="产品名称")
    price: float = Field(description="价格")
    category: str = Field(description="类别")

parser = JsonOutputParser(pydantic_object=ProductInfo)

product_prompt = ChatPromptTemplate.from_messages([
    ("system", "从用户描述中提取产品信息，以JSON格式返回。\n{format_instructions}"),
    ("user", "{description}")
])

# 注入格式说明
formatted = product_prompt.format_messages(
    format_instructions=parser.get_format_instructions(),
    description="一款售价299元的蓝牙耳机"
)

response = llm.invoke(formatted)
product = parser.parse(response.content)
print(f"产品：{product.name}, 价格：{product.price}")

# ============================
# 4. 对话管理
# ============================

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 存储聊天历史
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。"),
    ("placeholder", "{chat_history}"),
    ("human", "{input}")
])

chain = chat_prompt | llm | StrOutputParser()

# 带历史的链
with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# 多轮对话
session = {"configurable": {"session_id": "user_001"}}

reply1 = with_history.invoke({"input": "我叫张伟"}, config=session)
reply2 = with_history.invoke({"input": "我叫什么名字？"}, config=session)

print(reply1)
print(reply2)  # 应该记得"张伟"
```

## 版本说明

LangChain 发展迅速，目前已进入 **0.3.x** 稳定版本。关键版本差异：

```python
# 老式写法（langchain 0.1 之前，已废弃）
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain

# 新式写法（推荐，langchain >= 0.3）
from langchain_openai import ChatOpenAI       # 从子包导入
from langchain_core.prompts import ChatPromptTemplate  # core 是稳定基础

# LCEL（LangChain Expression Language）是标准的链构建方式
chain = prompt | llm | StrOutputParser()  # 管道语法

# LangChain 0.3 重要变化：
# - 完全移除了 langchain 0.1 的废弃 API
# - langchain-community 中的集成逐步迁移到独立包
# - 推荐配合 LangGraph 处理复杂 Agent 工作流
# - 内置 Pydantic V2 支持

# 检查版本
import langchain
print(langchain.__version__)  # 应为 0.3.x
```

---

## 小结

LangChain 的五大核心：模型、提示、输出解析、链、Agent。
推荐使用 LCEL 管道语法（`|` 符号连接），这是 LangChain 的未来方向。

---

*下一节：[11.2 Chain：构建处理管道](./02_chains.md)*

---

## 参考文献

[1] LangChain Team. LangChain Architecture Overview. https://python.langchain.com/docs/concepts/architecture, 2025.

[2] LangChain Team. Runnable Interface. https://python.langchain.com/docs/concepts/runnables, 2025.
