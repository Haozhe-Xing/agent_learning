# 附录 D：术语表

> 按字母顺序排列的 Agent 开发核心术语。

---

| 术语 | 英文 | 定义 |
|------|------|------|
| A2A 协议 | Agent-to-Agent Protocol | Google 提出的 Agent 间通信标准，让不同框架的 Agent 可以相互发现和调用 |
| AG2 | AG2 (AutoGen fork) | AutoGen 0.2 的社区分叉版本，由原 AutoGen 核心贡献者维护 |
| Agent | Agent / 智能体 | 能自主感知环境、做出决策并执行行动的 AI 系统 |
| AST | Abstract Syntax Tree / 抽象语法树 | 源代码的树形结构化表示，用于代码分析和理解 |
| Chain | Chain / 链 | 将多个 LLM 调用或处理步骤串联起来的序列 |
| CoT | Chain-of-Thought | 链式思维提示，引导模型逐步推理 |
| DeepSeek | DeepSeek | 深度求索，中国 AI 公司，推出 DeepSeek-R2 等高性能推理模型 |
| Docker | Docker | 容器化平台，将应用和依赖打包为可移植的容器镜像 |
| Embedding | Embedding / 嵌入 | 将文本转化为高维向量的过程 |
| FastAPI | FastAPI | Python 高性能异步 Web 框架，常用于 Agent API 服务化 |
| FastMCP | FastMCP | MCP Server 的简化创建方式，用装饰器快速定义工具 |
| Few-shot | Few-shot Learning | 通过少量示例引导模型完成任务 |
| Fine-tuning | Fine-tuning / 微调 | 在预训练模型基础上，用特定数据集进一步训练以适应特定任务 |
| Flows | Flows（CrewAI） | CrewAI 的事件驱动工作流编排特性，使用 @start/@listen/@router 装饰器 |
| Function Calling | Function Calling | LLM 生成结构化的工具调用请求的能力 |
| Graph Agent | Graph Agent / 图 Agent | 基于有向图结构构建的 Agent 工作流 |
| Hallucination | Hallucination / 幻觉 | LLM 生成看似合理但实际错误的内容 |
| Handoff | Handoff / 交接 | OpenAI Agents SDK 中 Agent 间转交控制权的机制 |
| Human-in-the-Loop | Human-in-the-Loop | 在 Agent 执行关键操作前请求人类确认 |
| LCEL | LangChain Expression Language | LangChain 的声明式链构建语法 |
| LLM | Large Language Model | 大语言模型，如 GPT-5、Claude 4、DeepSeek-R2、Llama 4 |
| MCP | Model Context Protocol | Anthropic 提出的模型与工具交互的标准协议 |
| Multi-Agent | Multi-Agent System | 多个 Agent 协作完成任务的系统 |
| OpenAI Agents SDK | OpenAI Agents SDK | OpenAI 推出的轻量 Agent 开发框架，Swarm 的生产级继承者 |
| PII | Personally Identifiable Information | 个人可识别信息，如姓名、身份证号、手机号等敏感数据 |
| Prompt | Prompt / 提示词 | 发送给 LLM 的输入文本 |
| Prompt Injection | Prompt Injection | 通过恶意输入覆盖 Agent 系统指令的攻击 |
| Pydantic | Pydantic | Python 数据验证库，常用于定义工具的输入/输出 Schema |
| RAG | Retrieval-Augmented Generation | 检索增强生成，先检索相关文档再生成回答 |
| ReAct | Reasoning + Acting | 推理与行动交替进行的 Agent 框架 |
| Reasoning Model | Reasoning Model / 推理模型 | 具备深度推理能力的 LLM，如 o3、DeepSeek-R2、Claude 4 Extended Thinking |
| Reflection | Reflection / 反思 | Agent 检查和纠正自己输出的机制 |
| Retriever | Retriever / 检索器 | 从知识库中检索相关文档的组件 |
| Runnable | Runnable / 可运行对象 | LangChain 中所有可执行组件的基础接口 |
| Sandbox | Sandbox / 沙箱 | 隔离的安全执行环境，防止恶意代码影响宿主系统 |
| Semantic Cache | Semantic Cache / 语义缓存 | 基于语义相似度（而非精确匹配）缓存 LLM 查询结果的技术 |
| SSE | Server-Sent Events | 服务器向客户端推送实时事件的协议 |
| State | State / 状态 | Agent 在执行过程中维护的上下文信息，包括对话历史、中间结果、工作记忆等，是 Graph Agent 中节点间数据传递的核心机制 |
| Streamable HTTP | Streamable HTTP | MCP 2025 年引入的新传输协议，替代 HTTP + SSE，支持按需流式和 Session 恢复 |
| Supervisor | Supervisor / 监督者 | 多 Agent 系统中协调其他 Agent 的中心节点 |
| System Prompt | System Prompt / 系统提示词 | 定义 Agent 行为准则的初始指令 |
| Temperature | Temperature / 温度 | 控制 LLM 输出随机性的参数（0=确定性，1=更随机） |
| Text-to-SQL | Text-to-SQL | 将自然语言描述自动转换为 SQL 查询语句的技术 |
| Token | Token / 令牌 | LLM 处理文本的最小单位（一个中文字约 1-2 个 Token） |
| Tool | Tool / 工具 | Agent 可以调用的外部功能（如搜索、计算、API 调用） |
| uv | uv | Rust 编写的高性能 Python 包管理器，已成为 Python 包管理新标准 |
| Vector DB | Vector Database / 向量数据库 | 存储和检索向量嵌入的专用数据库（如 ChromaDB、Pinecone） |
| Zero-shot | Zero-shot Learning | 不提供示例，仅通过指令让模型完成任务 |
