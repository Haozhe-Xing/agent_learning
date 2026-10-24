# 第6章 检索增强生成（RAG）

**阅读层次**：核心必读；进阶 RAG 按需选读。

## 本章迭代

- **上一版的不足**：任务步骤已受控，距离却可能来自模型记忆；命中了文档也不代表回答引用正确，空回答仍可能被误当作研究完成。
- **本章增加**：为同一任务保留命中文档、原文与来源，让提取步骤消费实际证据；分别记录检索、生成和验证状态，只把回答实际引用且验收通过的标签列为 `sources`。
- **怎样验证**：已有入口 `python -m reference_code.evidence_assistant.demo --stage rag`；运行 `python -m unittest reference_code.evidence_assistant.tests.test_answer_acceptance reference_code.evidence_assistant.tests.test_project_contracts` 检查空回答、未知引用、拒绝验收、资料变化和资料冲突。交付文档记录、回答及验收理由。问答模块默认只核对引用标签，不能证明结论被原文支持；语义支持需独立任务验收器或人工检查。
- **下一步**：文档和记忆增多后，关键距离可能被挤出输入。在[第7章](../chapter_context_engineering/README.md)把同一证据链纳入上下文预算；先保住必需资料，再考虑更复杂的检索后端。

RAG把检索到的外部资料交给回答阶段。它可以补充模型输入中的证据，但不会自动保证来源真实、回答正确或权限合规。本章先用本地小任务观察这些边界，再按需要选择向量与高级检索。

## 学习目标

完成核心练习后，应能：

- 区分检索候选、生成回答、引用检查、原文支撑和需求覆盖。
- 加载本地UTF-8文本/Markdown，按字符保真切块并保留来源版本与偏移。
- 用真实Chroma和固定向量检查距离、来源过滤、重开与删除。
- 通过BM25、同ID的RRF和重排提议比较排序机制，不把分数当正确概率。
- 运行带独立摘录/覆盖判据的词法问答，保留正常、未知与失败状态。

## 本章结构

| 小节 | 内容 | 难度 |
|---|---|---|
| 6.1 RAG的概念与工作原理 | 分开检索、回答与验收 | ⭐⭐ |
| 6.2 文档加载与文本分割 | 本地txt/md、保真字符切块与来源位置 | ⭐⭐ |
| 6.3 向量嵌入与向量数据库 | 显式嵌入合同、真实Chroma与距离 | ⭐⭐⭐ |
| 6.4 检索策略与重排序 | BM25/RRF/评分提议及反例 | ⭐⭐⭐ |
| 6.5 实战：智能文档问答Agent | 词法QA、原文摘录与独立覆盖 | ⭐⭐⭐ |
| 6.6 论文解读：RAG前沿进展 | Self-RAG/CRAG/GraphRAG等研究，选读 | ⭐⭐⭐ |
| 6.7 进阶RAG | 图检索与Agentic编排设计，按需选读 | ⭐⭐⭐⭐⭐ |

## 参考代码与完成条件

加载/分块机制在`reference_code/memory_rag/`，向量/混合检索在`reference_code/vector_retrieval/`，本地问答与综合调研应用复用`reference_code/evidence_assistant/`。这些练习职责不同；当前没有把向量候选直接接入QA的组合适配器。6.5明确说明来源映射、版本、命中快照与预算的接入合同。

引用标签存在不等于原文支持；有效摘录也可能缺必答点。交付需保留独立预期、实际来源、阶段状态和最终判定。真实语义模型、PDF/Word/Web加载、线上服务、GraphRAG与生产Agentic系统均不因本地练习通过而视为已验收。无需先安装所有高级依赖。

## 学习路径

前置知识：[第4章记忆](../chapter_memory/README.md)，以及Python文件、列表和字典操作。核心练习无需模型密钥或OpenAI嵌入API经验；Chroma机制练习使用锁定依赖。

接着在[第7章上下文工程](../chapter_context_engineering/README.md)管理检索证据的输入预算。需要复杂状态编排时，再读[第11章LangGraph](../chapter_langgraph/README.md)。

*下一节：[6.1 RAG的概念与工作原理](01_rag_concepts.md)*
