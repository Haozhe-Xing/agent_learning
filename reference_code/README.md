# 独立参考代码

这里存放可以按项目阅读、运行和测试的参考代码。正文位于 `src/`，章节的简短演示入口位于 `examples/`。

| 项目 | 用途 | 从仓库根目录运行 |
|---|---|---|
| [本地资料调研助手](evidence_assistant/README.md) | 中等难度的单 Agent 项目：多份本地资料、多问题检索、上下文预算和带来源的报告 | `python -m reference_code.evidence_assistant` |
| 同项目中的基础迭代 | 用受控距离资料分别观察工具、记忆、规划、RAG、上下文与 Harness | `python -m reference_code.evidence_assistant.demo --all` |
| [受限SQLite连接](sqlite_readonly.py) | 第22.2节的可运行数据库边界与模型提议接口 | `python -m unittest discover -s reference_code/tests -v` |
| [记忆与分块练习](memory_rag/README.md) | 第4章及6.1–6.2的历史预算、任务状态、同意存储与保真分块 | `python -m reference_code.memory_rag` |
| [向量与融合检索练习](vector_retrieval/README.md) | 第6.3–6.4的真实Chroma、BM25、RRF及重排提议校验 | `python -m reference_code.vector_retrieval` |
| [LangChain数据流练习](langchain_patterns/README.md) | 第10章的组合、校验、本地追踪及有界重试 | `python -m reference_code.langchain_patterns` |
| [LangGraph生命周期练习](langgraph_patterns/README.md) | 第11章的reducer、有限修订、动态审批与本地收据 | `python -m reference_code.langgraph_patterns` |
| [工具调用协议练习](tool_calling/README.md) | 第3章的本地搜索/受限计算、工具ID、预算及冻结结果验收 | `python -m reference_code.tool_calling` |
| [规划与修订练习](planning_patterns/README.md) | 第5章的有界行动、DAG依赖、来源/目标独立验收 | `python -m reference_code.planning_patterns` |
| [授权边界练习](security_boundary/README.md) | 第19章的宿主许可、内容绑定审批、本地收据与正常/越权回归 | `python -m reference_code.security_boundary` |

主项目的实现、样例资料、独立测试和数据契约放在其项目目录内，使用Python标准库与离线资料。机制练习各有目录；工具/规划练习按README复用仓库已有模块，授权边界目录可单独复制。框架/Chroma依赖锁定在`examples/requirements.txt`，相应测试放在`examples/integration_tests/`；各README说明范围与命令。先跑通主项目，再按章节选择练习，避免同时引入全部框架和扩展。

`reference-agent/` 保留为另外的协议与服务参考底座；它与本目录的项目各有职责，入口见其 [README](../reference-agent/README.md)。

## English

Start with the [local evidence research assistant](evidence_assistant/README.md), which keeps its implementation, sample data, tests, and contracts together. It answers multiple questions from local documents and produces a cited report under explicit execution and context budgets. The smaller six-stage fixture remains a separate introductory exercise.

Optional mechanism labs live in separate directories: [memory and chunking](memory_rag/README.md), [vector retrieval](vector_retrieval/README.md), [LangChain](langchain_patterns/README.md), and [LangGraph](langgraph_patterns/README.md). Their framework/Chroma dependencies are pinned in `examples/requirements.txt`; integration tests remain in `examples/integration_tests/`. Each lab declares its fixture and validation boundaries. Additional standard-library labs cover [tool calling](tool_calling/README.md), [planning](planning_patterns/README.md) and [authorization boundaries](security_boundary/README.md). Tool/planning labs reuse declared repository modules; the authorization lab can be copied on its own. Choose labs by chapter rather than adding every dependency to the main project.
