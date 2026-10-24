# 配套示例与运行边界

使用 Python 3.11+，从仓库根目录运行。默认入口使用虚构资料和确定性测试结果，便于复现执行行为；真实模型模式需显式传入 `--real`。模型训练、研究扩展与外部业务后端不属于这些最小应用的已验证范围。

## 独立参考项目

完整参考代码已经集中到 [`reference_code/evidence_assistant/`](../reference_code/evidence_assistant/README.md)，包括实现、样例资料、测试和契约。主体验是多文档、多问题的本地资料调研助手：

```bash
python -m reference_code.evidence_assistant
```

先运行并替换资料、问题，再逐步学习检索、计划、上下文与执行约束。自定义资料/问题需配置独立的`--task-spec`；没有本题要求时，即使摘录有效也保持覆盖未知，CLI返回1。项目目录内也可用 `python run.py`。目录、任务规格示例与难度安排见项目 README，实际输出讲解见 [WALKTHROUGH.md](../reference_code/evidence_assistant/WALKTHROUGH.md)。

## 六阶段基础练习

```bash
python -m reference_code.evidence_assistant.demo --all
python -m reference_code.evidence_assistant.demo --stage harness
python -m reference_code.evidence_assistant.demo --all --output-unit milliseconds
```

同一任务依次加入工具、记忆、规划、检索、上下文与执行约束。资料里的数量和单位决定计算结果；`--file PATH` 可替换资料，`--output-unit` 会进入记忆与换算。代码位于 `reference_code/evidence_assistant/`。示例目录保留最初的编号，书稿章节使用新目录编号，例如第10章的入口仍是 `examples/chapter12/customer_service.py`。输入格式、完成判据和独立验收见项目 [CONTRACTS.md](../reference_code/evidence_assistant/CONTRACTS.md)，完整阅读路线见书稿的“贯穿项目与实战验收”页。CLI 仅在全部结果验收通过时返回 `0`。

## 配套入口

| 章节 | 入口 | 默认行为与运行条件 |
|---|---|---|
| 3.5 | `examples/chapter03/search_calc_agent.py --demo` | 标准库离线计算与换算；不加 `--demo` 时调用真实模型与搜索，需 API Key |
| 4.1–4.4、6.1–6.2 | `-m reference_code.memory_rag` | 字节有界历史、任务状态、保真分块与真实Chroma固定向量；独立参考目录 |
| 4.3 | `examples/chapter04/long_term_memory.py` | SQLite 磁盘写入、重开、读取与删除；示例使用临时目录 |
| 4.4 | `examples/chapter04/scratchpad_agent.py` | 验证工作记忆容量与淘汰；不模拟模型推理 |
| 4.5 | `examples/chapter04/personal_assistant.py` | 真实 ChromaDB 与模型 API；需安装依赖和 API Key，持久写入默认关闭 |
| 4.7 | `examples/chapter04/layered_memory_agent.py` | Core、Working、Archive 的独立分层基线 |
| 5.5 | `examples/chapter05/research_agent.py` | 授权本地资料上的证据报告；`--file PATH` 指定资料，`--real` 开启模型回答 |
| 5.6 | `examples/chapter05/plan_and_execute.py` | 工具失败后替换执行队列，返回步骤轨迹与状态 |
| 6.5 | `examples/chapter06/doc_qa_agent.py` | 关键词检索与来源引用；`--real` 开启模型回答；不是 Chroma 向量检索 |
| 6.3–6.4 | `-m reference_code.vector_retrieval` | 真实Chroma、BM25与RRF、校验重排提议；固定向量不证明语义质量 |
| 6.7 | `examples/chapter06/agentic_rag.py` | 有次数上限的查询重试基线；不包含 GraphRAG 后端 |
| 7.4 | `examples/chapter07/gssc_pipeline.py` | 完整消息的字节预算、去重、截断和取舍记录 |
| 8.5 | `examples/chapter08/harness_demo.py` | 宿主强制检查工具许可、调用预算和产物验收 |
| 10.3 | `-m examples.chapter12.first_agent` | 真实 LangChain `create_agent`、受限计算工具与固定模型；可加 `--real` |
| 10.1–10.8 | `-m reference_code.langchain_patterns` | Runnable、局部追踪、JSON校验、选择性重试、流式错误与缓存边界；无远程追踪 |
| 10.5 | `examples/chapter12/customer_service.py` | 真实 LangChain 工具循环，默认固定模型；可加 `--real` |
| 11.3 | `-m examples.chapter13.first_graph_agent` | 真实 LangGraph 工具节点与条件路由；可加 `--stream` 或 `--real` |
| 11.2、11.4–11.5 | `-m reference_code.langgraph_patterns` | 默认并发检查；另选`review`、`approval`观察修订/动态审批与本地收据 |
| 11.6 | `examples/chapter13/content_workflow.py` | 真实 LangGraph 状态图，默认固定评分；可加 `--real` 或 `--always-low` |
| 25 | `examples/auto_research_demo/main.py` | 标准库候选搜索与固定评估；不是网页研究助手 |
| 16.5 | `examples/dev_team/dev_team.py` | 按同目录配置安装；会执行生成程序，应在隔离环境运行 |
| 17–20 | `reference-agent/` | 最小工具、评估、基础防护、MCP 与 HTTP 服务；见其 README |
| 22.2 | `reference_code/sqlite_readonly.py` | 标准库只读SQLite与SQL提议适配器；按正文创建虚构数据库或运行下列测试 |

表中的脚本参数附在路径后运行，例如 `python examples/chapter03/search_calc_agent.py --demo`；`-m`条目按模块运行，例如 `python -m examples.chapter12.first_agent`。22.2是被导入的模块，复现命令见正文。

## 安装与验证

```bash
python -m venv .venv
# 激活该虚拟环境后执行以下命令
pip install -r examples/requirements.txt
python -m unittest discover -s reference_code/evidence_assistant/tests -v
python -m unittest discover -s reference_code/tests -v
python -m unittest discover -s examples/tests -v
python -m pytest examples/integration_tests -q
PYTHONPATH=examples/dev_team python -m pytest examples/dev_team/tests -q
PYTHONPATH=reference-agent/src python -m pytest reference-agent/tests -q
python tools/verify_examples.py --integration --report /tmp/agent-examples.json
python tools/check_book.py
python tools/check_learning_path.py
python tools/check_snippets.py
python tools/build_books.py
```

`requirements.txt` 固定本轮验证的直接依赖版本。标准库测试不需要安装全部依赖。集成测试使用真实 ChromaDB、LangChain、LangGraph 和 OpenAI SDK；向量与模型响应固定，不访问线上模型。

第4.5节的记忆入口提供 `memories` 查看、`forget` 删除当前用户记忆并清空会话、`reset` 仅清空会话，以及 `exit`。只有加 `--remember` 才开启自动持久写入；此前已保存的记忆仍可读取。请使用虚构身份；生产身份必须由服务端鉴权。数据默认在运行目录 `data/` 下。

真实模型示例使用 `OPENAI_API_KEY`，可通过 SDK 支持的 `OPENAI_BASE_URL` 接入兼容服务。模型名由 `--model` 指定。Key 缺失时不要把离线结果当成线上验证；准确性、语义召回与成本需另用真实任务集测量。

验证结果和边界见 `examples/VALIDATION.md`。构建工具版本见 `tools/README.md`。

## Companion Examples in English

Run the self-contained local research project with `python -m reference_code.evidence_assistant`; its implementation, sample documents, and tests live in `reference_code/evidence_assistant/`. Run from the repository root with Python 3.11+. The table lists chapter-specific entry points. `python -m reference_code.evidence_assistant.demo --all` runs the shared offline fixture across six stages. Framework examples execute real tool loops and state graphs with deterministic outputs by default. Only `--real` invokes a live model, except the chapter 3 interactive and chapter 4.5 memory assistants, which explicitly require credentials.

Custom research documents/questions need an independent `--task-spec`. Without requirements for that question, valid excerpts leave coverage unknown and the CLI exits 1. See the project README for a complete example and [WALKTHROUGH.md](../reference_code/evidence_assistant/WALKTHROUGH.md) for actual report fields and failure paths. Sections 10.3 and 11.3 use the `python -m` module entries shown in the table; 22.2 supplies an importable SQLite module and real-database tests.

The optional `memory_rag`, `vector_retrieval`, `langchain_patterns`, and `langgraph_patterns` labs each live separately under `reference_code/`; run them with `python -m reference_code.NAME` as listed above. They expose mechanisms with local fixtures and real storage/frameworks, not live-model quality or complete production systems. Start with the main project and select labs by chapter.

Use `--file PATH` to replace the distance evidence and `--output-unit milliseconds` to exercise memory-driven output conversion. The CLI returns `0` only when every result is accepted. [project contracts](../reference_code/evidence_assistant/CONTRACTS.md) define the input grammar, data flow, independent checks and failure states.

Install pinned direct dependencies with `pip install -r examples/requirements.txt`. Unit tests validate standard-library behavior and substitutes; integration tests use actual storage and frameworks with fixed vectors and responses. These tests do not establish live-model factuality, semantic retrieval quality, or production security.

Memory persistence is opt-in with `--remember`; previously stored memories remain readable. Inspect with `memories`, delete the current user's records with `forget`, and clear conversation history with `reset`. Use synthetic identities. Production authentication and backup retention are separate responsibilities. See `examples/VALIDATION.md` for measured results and `tools/README.md` for the matched build toolchain.

旧的 `examples.learning_path` 仅保留兼容转发，未保留第二份实现；新示例与书稿统一使用独立参考项目。

## 工具、规划与安全核心修订（2026-10-22）

新增机制均位于独立参考目录，核心项目难度不变：

| 配套 | 默认运行 | 验证边界 |
|---|---|---|
| `reference_code/tool_calling/` | `python -m reference_code.tool_calling` | 固定三轮本地搜索/计算，闭合协议与独立冻结结果；SDK只走HTTP替身 |
| `reference_code/planning_patterns/` | `python -m reference_code.planning_patterns` | 来源与目标检查、DAG依赖和有界修订，可信本地回调 |
| `reference_code/security_boundary/` | `python -m reference_code.security_boundary` | 虚构内存读/公告、授权与审批、正常/越权效果；无真正发送与OS沙箱 |

测试分别在`examples/integration_tests/test_{tool_calling,planning,security}_chapter.py`。所有命令从仓库根目录运行；工具/规划复用已声明的仓库模块，安全目录可单独复制。没有新增第三方依赖或自动线上模式。以最终`VALIDATION.md`记录的计数和运行范围为准。
