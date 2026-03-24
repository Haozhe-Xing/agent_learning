# 附录 D：术语表

> 按字母顺序排列的 Agent 开发核心术语。

![Agent 核心术语关系图](../svg/appendix_glossary.svg)

---

| 术语 | 英文 | 定义 |
|------|------|------|
| A2A 协议 | Agent-to-Agent Protocol | Google 提出的 Agent 间通信标准，让不同框架的 Agent 可以相互发现和调用 |
| AG2 | AG2 (AutoGen fork) | AutoGen 0.2 的社区分叉版本，由原 AutoGen 核心贡献者维护 |
| Agent | Agent / 智能体 | 能自主感知环境、做出决策并执行行动的 AI 系统 |
| Agentic-RL | Agentic Reinforcement Learning | 将强化学习应用于 Agent 训练的范式，通过奖励信号让 Agent 学会更好地使用工具和规划 |
| ANP | Agent Network Protocol | Agent 网络协议，定义开放互联网中 Agent 间的发现和通信标准 |
| AST | Abstract Syntax Tree / 抽象语法树 | 源代码的树形结构化表示，用于代码分析和理解 |
| Attention | Attention Mechanism / 注意力机制 | Transformer 的核心机制，让模型在处理每个 Token 时关注输入序列中最相关的位置 |
| Chain | Chain / 链 | 将多个 LLM 调用或处理步骤串联起来的序列 |
| Checkpoint | Checkpoint / 检查点 | LangGraph 中保存图执行状态快照的机制，支持暂停和恢复 |
| Chunking | Text Chunking / 文本分块 | 将长文档切分为适合 Embedding 的小段落的过程 |
| Context Engineering | Context Engineering / 上下文工程 | 系统性地管理和优化 Agent 上下文信息的方法论，是 Prompt Engineering 的进阶 |
| Context Window | Context Window / 上下文窗口 | LLM 单次推理能处理的最大 Token 数量 |
| CoT | Chain-of-Thought | 链式思维提示，引导模型逐步推理 |
| CRAG | Corrective RAG | 在检索后添加纠错机制的 RAG 变体，能判断检索结果是否相关并动态调整策略 |
| CrewAI | CrewAI | 角色扮演型多 Agent 框架，通过定义 Agent 角色和任务来组建虚拟团队 |
| DeepSeek | DeepSeek | 深度求索，中国 AI 公司，推出 DeepSeek-R2 等高性能推理模型 |
| Dify | Dify | 开源的 LLM 应用开发平台，支持低代码构建 Agent 和 RAG 流程 |
| Docker | Docker | 容器化平台，将应用和依赖打包为可移植的容器镜像 |
| DPO | Direct Preference Optimization | 直接偏好优化，不需要训练奖励模型就能对齐人类偏好的训练方法 |
| Embedding | Embedding / 嵌入 | 将文本转化为高维向量的过程 |
| Emergent Abilities | Emergent Abilities / 涌现能力 | 大模型在参数规模达到临界点后突然涌现的新能力（如推理、代码生成） |
| FastAPI | FastAPI | Python 高性能异步 Web 框架，常用于 Agent API 服务化 |
| FastMCP | FastMCP | MCP Server 的简化创建方式，用装饰器快速定义工具 |
| Few-shot | Few-shot Learning | 通过少量示例引导模型完成任务 |
| Fine-tuning | Fine-tuning / 微调 | 在预训练模型基础上，用特定数据集进一步训练以适应特定任务 |
| Flows | Flows（CrewAI） | CrewAI 的事件驱动工作流编排特性，使用 @start/@listen/@router 装饰器 |
| Function Calling | Function Calling | LLM 生成结构化的工具调用请求的能力 |
| GAIA | GAIA Benchmark | General AI Assistants 基准测试，评估 Agent 解决真实世界任务的能力 |
| Graph Agent | Graph Agent / 图 Agent | 基于有向图结构构建的 Agent 工作流 |
| GRPO | Group Relative Policy Optimization | 组内相对策略优化，DeepSeek 提出的强化学习算法，不需要 Critic 模型 |
| Guardrails | Guardrails / 防护栏 | 对 Agent 输入输出进行安全检查和限制的机制 |
| Hallucination | Hallucination / 幻觉 | LLM 生成看似合理但实际错误的内容 |
| Handoff | Handoff / 交接 | OpenAI Agents SDK 中 Agent 间转交控制权的机制 |
| Human-in-the-Loop | Human-in-the-Loop | 在 Agent 执行关键操作前请求人类确认 |
| LCEL | LangChain Expression Language | LangChain 的声明式链构建语法 |
| LLM | Large Language Model | 大语言模型，如 GPT-5、Claude 4、DeepSeek-R2、Llama 4 |
| Long-term Memory | Long-term Memory / 长期记忆 | 持久化存储的 Agent 记忆，跨会话保留用户偏好和重要信息 |
| LoRA | Low-Rank Adaptation | 低秩适配，一种参数高效的微调方法，只训练少量额外参数 |
| MCP | Model Context Protocol | Anthropic 提出的模型与工具交互的标准协议 |
| Mermaid | Mermaid | 基于文本的图表绘制工具，用代码生成流程图、时序图等 |
| Multi-Agent | Multi-Agent System | 多个 Agent 协作完成任务的系统 |
| OpenAI Agents SDK | OpenAI Agents SDK | OpenAI 推出的轻量 Agent 开发框架，Swarm 的生产级继承者 |
| OpenTelemetry | OpenTelemetry / OTel | 开源的可观测性框架，用于收集 Agent 的追踪、指标和日志数据 |
| PII | Personally Identifiable Information | 个人可识别信息，如姓名、身份证号、手机号等敏感数据 |
| PPO | Proximal Policy Optimization | 近端策略优化，经典的强化学习算法，用于 RLHF 训练 |
| Prompt | Prompt / 提示词 | 发送给 LLM 的输入文本 |
| Prompt Injection | Prompt Injection | 通过恶意输入覆盖 Agent 系统指令的攻击 |
| Pydantic | Pydantic | Python 数据验证库，常用于定义工具的输入/输出 Schema |
| RAG | Retrieval-Augmented Generation | 检索增强生成，先检索相关文档再生成回答 |
| ReAct | Reasoning + Acting | 推理与行动交替进行的 Agent 框架 |
| Reasoning Model | Reasoning Model / 推理模型 | 具备深度推理能力的 LLM，如 o3、DeepSeek-R2、Claude 4 Extended Thinking |
| Reducer | Reducer（LangGraph） | LangGraph 状态管理中的聚合函数，定义多个节点输出如何合并到同一状态字段 |
| Reflection | Reflection / 反思 | Agent 检查和纠正自己输出的机制 |
| Retriever | Retriever / 检索器 | 从知识库中检索相关文档的组件 |
| RLHF | Reinforcement Learning from Human Feedback | 基于人类反馈的强化学习，用于对齐模型输出与人类偏好 |
| Runnable | Runnable / 可运行对象 | LangChain 中所有可执行组件的基础接口 |
| Sandbox | Sandbox / 沙箱 | 隔离的安全执行环境，防止恶意代码影响宿主系统 |
| Scaling Laws | Scaling Laws / 缩放定律 | 模型性能与参数量、数据量、计算量之间的幂律关系 |
| Scratchpad | Scratchpad / 草稿本 | Agent 的工作记忆空间，用于存放中间推理步骤 |
| Semantic Cache | Semantic Cache / 语义缓存 | 基于语义相似度（而非精确匹配）缓存 LLM 查询结果的技术 |
| SFT | Supervised Fine-Tuning | 监督微调，使用标注数据进行的模型微调 |
| Short-term Memory | Short-term Memory / 短期记忆 | 当前会话的对话历史，受上下文窗口限制 |
| Skill | Skill / 技能 | 封装了特定能力的可复用 Agent 组件，包含工具、提示词和执行逻辑 |
| SSE | Server-Sent Events | 服务器向客户端推送实时事件的协议 |
| State | State / 状态 | Agent 在执行过程中维护的上下文信息，是 Graph Agent 中节点间数据传递的核心机制 |
| Streamable HTTP | Streamable HTTP | MCP 2025 年引入的新传输协议，支持按需流式和 Session 恢复 |
| Supervisor | Supervisor / 监督者 | 多 Agent 系统中协调其他 Agent 的中心节点 |
| SWE-bench | SWE-bench | Software Engineering 基准测试，评估 Agent 解决真实 GitHub Issue 的能力 |
| System Prompt | System Prompt / 系统提示词 | 定义 Agent 行为准则的初始指令 |
| Temperature | Temperature / 温度 | 控制 LLM 输出随机性的参数（0=确定性，1=更随机） |
| Text-to-SQL | Text-to-SQL | 将自然语言描述自动转换为 SQL 查询语句的技术 |
| Token | Token / 令牌 | LLM 处理文本的最小单位（一个中文字约 1-2 个 Token） |
| Tool | Tool / 工具 | Agent 可以调用的外部功能（如搜索、计算、API 调用） |
| Transformer | Transformer | 现代 LLM 的核心架构，基于自注意力机制，由 Google 2017 年提出 |
| uv | uv | Rust 编写的高性能 Python 包管理器，已成为 Python 包管理新标准 |
| Vector DB | Vector Database / 向量数据库 | 存储和检索向量嵌入的专用数据库（如 ChromaDB、Pinecone） |
| Working Memory | Working Memory / 工作记忆 | Agent 执行任务时的临时推理空间，类似于人类的"草稿纸" |
| Zero-shot | Zero-shot Learning | 不提供示例，仅通过指令让模型完成任务 |
