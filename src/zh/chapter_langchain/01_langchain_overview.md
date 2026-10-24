# 12.1 LangChain 架构全景

LangChain 是一个模块化的 LLM 应用开发框架，核心设计思想是**通过标准化接口组合各类组件**。但"为什么需要这一层抽象"比"它有哪些组件"更值得先想清楚。

## 为什么需要抽象层：从手搓到框架

在本书前半部分，你已经从零手搓过 Agent——自己管理消息列表、自己写工具 Schema、自己跑执行循环。那时你拥有完全的控制力，也切身体会到"样板代码"的重量。LangChain 解决的正是一个反复出现的工程问题：

> **把"调用 LLM → 处理输入 → 调用工具 → 解析输出"这类横切关注点，收敛成可复用的标准组件。**

| 维度 | 从零手搓 | 使用 LangChain |
|------|---------|---------------|
| 接入新模型/工具 | 写适配代码 | 换一个 import，接口一致 |
| 流式/异步/批量 | 自己实现 | 组件天然支持 |
| 链路可观测 | 手动打点 | 统一回调 + LangSmith |
| 组合复杂度 | 胶水代码随规模膨胀 | `\|` 管道声明式组合 |
| 抽象泄漏时 | 无（没有抽象） | 需钻出框架定制 |

> 📌 **关键判断**：框架的价值不在"让你好奇地探索"，而在"让你重复造的轮子消失"。当你清楚自己要什么、且样板代码已成为负担时，框架的杠杆才正向。本书前面的手搓章节，正是为了让你**先理解原理，再借框架加速**。

## 核心组件体系

![LangChain核心组件体系](../svg/chapter_langchain_01_langchain_components.svg)

LangChain 的组件可以归为五类：**模型（Models）**、**提示（Prompts）**、**输出解析（Output Parsers）**、**链（Chains）**、**Agent**。理解这五类及其关系，就掌握了 LangChain 的骨架。

## 架构演进：从单体到分包

LangChain 自 2022 年底发布以来经历了三次重大架构变革，理解这些变化有助于你读懂网上不同时期的教程，避免踩「API 已废弃」的坑。

| 阶段 | 版本 | 核心变化 | 关键特征 |
|------|------|---------|---------|
| **v0.0.x（2022.11—2023.12）** | 单体包 | 所有功能在一个 `langchain` 包中 | `from langchain.llms import OpenAI` |
| **v0.1.x（2024.01—2024.06）** | 分包架构 | 拆分出 `langchain-core`、`langchain-community` | 双重导入路径并存，旧 API 标记 deprecated |
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
- **集成包**：每个 LLM 提供商一个包，各自独立发版，不会因为某个提供商的 API 变更影响其他用户。

> 💡 **实践建议**：新项目一律使用 `from langchain_openai import ChatOpenAI` 这样的分包导入，不要使用 `from langchain.chat_models import ChatOpenAI`（这是旧路径的兼容别名，已标记废弃）。

### Runnable 协议：LangChain 的"统一语言"

Runnable 是 LangChain 0.2+ 引入的**最核心抽象**——所有组件（LLM、Prompt、Parser、Tool、Retriever）都实现了统一的 Runnable 接口 [2]。这意味着它们共享一套完全一致的调用方式：

```python
from langchain_core.runnables import Runnable

# 所有 Runnable 都支持：invoke / ainvoke / stream / astream / batch / abatch
chain = prompt | llm | parser          # 用 | 管道符连接（即 LCEL）
parallel = RunnableParallel(              # 同一个输入同时执行多条链
    summary=summary_chain, translation=translate_chain
)
```

**为什么 Runnable 协议如此重要？**

1. **可组合性**：任何 Runnable 都可以用 `|` 串联或用 `RunnableParallel` 并联，构建任意复杂的处理管道
2. **流式默认**：`.stream()` 让所有组件都支持流式输出——对需要实时返回的 Agent 应用至关重要
3. **可观测性**：内置回调系统（`callbacks`）可以追踪每个 Runnable 的输入输出，配合 LangSmith 实现全链路监控
4. **类型安全**：每个 Runnable 有明确的 `input_schema` 和 `output_schema`（基于 Pydantic）

> 💡 **直觉理解**：把 Runnable 想象成"带标准接口的乐高积木"——不管这块积木是调模型还是查数据库，它对外暴露的"插口"（`invoke`/`stream`/`batch`）都一样。于是你可以用 `|` 把任意两块拼起来，而无需关心内部实现。这正是"抽象"的力量：**统一接口让组合成为可能**。

### LangChain 与同类框架对比

在选择开发框架时，了解不同框架的定位有助于做出合理选择（更系统的选型决策见 [第三部分总览](../part3.md)）：

| 框架 | 核心定位 | 最适合场景 | 社区活跃度 |
|------|---------|-----------|-----------|
| **LangChain** | 通用 LLM 编排 | 需要大量集成的企业应用 | ⭐⭐⭐⭐⭐ |
| **LlamaIndex** | 数据连接 + RAG | 文档问答、知识库 | ⭐⭐⭐⭐ |
| **Haystack** | 搜索 + RAG Pipeline | 搜索增强型应用 | ⭐⭐⭐ |
| **Semantic Kernel** | 微软生态集成 | Azure + C# 项目 | ⭐⭐⭐ |
| **原生 API** | 无框架依赖 | 简单原型、极致性能 | — |

> 📌 **选择建议**：如果你需要快速接入多种 LLM 和工具，LangChain 的集成生态是最大优势；如果场景以 RAG 为核心，LlamaIndex 更专精；如果追求极致可控性，直接使用原生 API。在本书实战项目中，我们选择 LangChain + LangGraph 组合，因为它在编排灵活性和社区支持上最均衡。

---

## 快速上手（最小片段）

下面的片段展示 LangChain 的"最小可运行骨架"——后续所有能力都建立在这三行之上：

```python
# pip install langchain langchain-openai langchain-community
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0.7)
chain = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专注于{domain}领域。"),
    ("user", "{question}")
]) | llm | StrOutputParser()

print(chain.invoke({"role": "Python 专家", "domain": "机器学习",
                    "question": "如何用 sklearn 训练一个分类器？"}))
```

> 📌 版本说明：LangChain 已进入 **0.3.x** 稳定版，推荐使用分包导入（`langchain_openai`、`langchain_core`）。`LLMChain` 等旧式链已被移除，**LCEL（`prompt | llm | parser`）是标准的链构建方式**。新项目构建 Agent 也建议直接用 LangGraph（见第 13 章）。

---

## 小结

LangChain 的五大核心：模型、提示、输出解析、链、Agent。其真正的内核是 **Runnable 统一协议** + **分层包结构**——前者让组件可自由组合，后者让"稳定接口"与"易变实现"解耦。推荐使用 LCEL 管道语法（`|` 符号），这是 LangChain 的未来方向。

---

*下一节：[12.2 Chain：构建处理管道](./02_chains.md)*

---

## 参考文献

[1] LangChain Team. LangChain Architecture Overview. https://python.langchain.com/docs/concepts/architecture, 2025.

[2] LangChain Team. Runnable Interface. https://python.langchain.com/docs/concepts/runnables, 2025.
