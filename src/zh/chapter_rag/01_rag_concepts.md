# 7.1 RAG 的概念与工作原理

RAG（Retrieval-Augmented Generation）是一种将**信息检索**与**语言生成**结合的技术架构。它让 LLM 能够基于外部知识库回答问题，而不是仅依赖训练时学到的知识。

> 📄 **论文出处**：RAG 的概念由 Meta AI（当时的 Facebook AI Research）在论文 *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"*（Lewis et al., 2020）中首次提出。原始论文将检索模型（DPR）和生成模型（BART）端到端联合训练，在开放域问答任务上大幅超越了传统的"先检索后阅读"方案。虽然今天的 RAG 实现方式已经与原始论文有很大不同（我们通常不做端到端训练，而是将检索和生成解耦），但核心思想完全一致：**让模型在生成回答时能够参考外部知识。**

## 为什么需要 RAG？

LLM 有三个根本局限：

```python
# 局限 1：知识截止日期
question = "最新版本的 GPT-4 有哪些新功能？"
# LLM 只知道训练数据截止前的信息

# 局限 2：领域知识匮乏
question = "我们公司的退款政策是什么？"
# LLM 不知道你公司的内部文档

# 局限 3：幻觉风险
question = "张三博士2023年发表了哪些论文？"
# LLM 可能编造不存在的论文
```

RAG 通过"先检索、再生成"解决这三个问题：
- ✅ 检索最新文档 → 解决知识截止
- ✅ 检索内部知识库 → 解决领域匮乏
- ✅ 基于真实文档生成 → 减少幻觉

## RAG 工作流程

![RAG 工作流程](../svg/chapter_rag_01_rag_flow.svg)

> 🎬 **交互式动画**：完整体验 RAG 的五步流水线——从文档分块、向量化、向量空间搜索可视化，到 Prompt 组装和 Token 流式生成，每一步都可以交互操作。
>
> <a href="../animations/rag_flow.html" target="_blank" style="display:inline-block;padding:8px 16px;background:#FF9800;color:white;border-radius:6px;text-decoration:none;font-weight:bold;">▶ 打开 RAG 工作流程交互动画</a>

## 核心概念

### 1. Chunk（文本块）

文档被分割为一段段的文本块（Chunk），每个 Chunk 独立存储和检索。

```python
# 一个文档可能被分成很多 Chunk
document = "这是一篇关于Python的长文章..." 

chunks = [
    "Python 是由 Guido van Rossum 创建的...",     # Chunk 1
    "Python 的设计哲学强调代码的可读性...",          # Chunk 2
    "Python 在人工智能领域应用广泛，包括...",         # Chunk 3
    # ...
]
```

### 2. Embedding（向量嵌入）

将文本转化为可比较的数字向量：

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    """将文本转为 1536 维向量"""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# 语义相似的文本有相近的向量
v1 = embed("Python 编程语言")
v2 = embed("Python 是用于编程的工具")
v3 = embed("今天天气不错")

# v1 和 v2 的余弦相似度 > 0.9
# v1 和 v3 的余弦相似度 < 0.5
```

### 3. 相似度检索

```python
import chromadb
import numpy as np

# 用余弦相似度找最相关的文档块
def find_relevant_chunks(query: str, collection, n: int = 5) -> list[str]:
    """从向量库中找最相关的文档块"""
    
    query_embedding = embed(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "distances"]
    )
    
    chunks = results["documents"][0]
    distances = results["distances"][0]
    
    # 返回并打印相关性
    for chunk, dist in zip(chunks, distances):
        similarity = 1 - dist  # 转换为相似度
        print(f"[{similarity:.2f}] {chunk[:80]}...")
    
    return chunks
```

### 4. 上下文注入（Context Injection）

将检索到的相关文档块注入到 Prompt 中：

```python
def answer_with_context(question: str, context_chunks: list[str]) -> str:
    """基于上下文回答问题"""
    
    # 构建上下文字符串
    context = "\n\n---\n\n".join(context_chunks)
    
    prompt = f"""请基于以下参考资料回答问题。
    
【参考资料】
{context}

【问题】
{question}

【要求】
- 只使用参考资料中的信息
- 如果资料中没有相关信息，明确说明
- 引用具体的信息来支持你的回答
"""
    
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# 完整流程
question = "Python 是什么时候创建的？"
relevant_chunks = find_relevant_chunks(question, collection)
answer = answer_with_context(question, relevant_chunks)
print(answer)
```

## RAG vs 直接查询 LLM

![RAG vs 直接查询LLM对比](../svg/chapter_rag_01_rag_vs_llm.svg)

```python
def compare_approaches(question: str, has_relevant_docs: bool = True):
    """对比 RAG 和直接查询的效果"""
    
    # 方式1：直接问 LLM（可能幻觉）
    direct_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": question}]
    )
    
    print("=== 直接问 LLM ===")
    print(direct_response.choices[0].message.content[:300])
    
    # 方式2：RAG（基于文档）
    if has_relevant_docs:
        chunks = find_relevant_chunks(question, collection)
        rag_answer = answer_with_context(question, chunks)
        
        print("\n=== RAG 回答 ===")
        print(rag_answer[:300])

# 特别适合内部知识库查询
compare_approaches("我们公司的产品退款流程是什么？")
```

---

## 小结

RAG 的核心价值：
- **解决知识局限**：外挂任意知识库
- **减少幻觉**：基于真实文档生成
- **实时更新**：更新文档即更新知识
- **可溯源**：每个回答都有文档依据

### RAG 的局限与挑战

RAG 不是万能的，在实际应用中你可能会遇到以下挑战：

| 挑战 | 描述 | 应对策略 |
|------|------|---------|
| **检索质量瓶颈** | 如果检索到了无关文档，再好的 LLM 也无法生成正确答案 | 优化嵌入模型、使用混合检索、重排序（详见 7.4 节） |
| **长上下文稀释** | 检索到太多文档会"稀释"关键信息，反而降低回答质量 | 控制检索数量（top_k）、压缩文档摘要 |
| **跨文档推理困难** | 答案分散在多个文档中时，LLM 难以有效整合 | 使用 Map-Reduce 策略、分步骤推理 |
| **数据新鲜度** | 向量索引需要定期更新，否则包含过时信息 | 设计增量更新机制、添加时间戳过滤 |
| **结构化数据不友好** | RAG 对表格、数据库等结构化数据的处理效果不如非结构化文本 | 结合 Text-to-SQL 方案（详见第 20 章） |

理解这些局限有助于你在实际项目中合理评估 RAG 的适用范围，并选择正确的优化方向。

> 📖 **想深入了解 RAG 的学术前沿？** 请阅读 [9.6 论文解读：RAG 前沿进展](./06_paper_readings.md)，涵盖 RAG 原始论文、Self-RAG、CRAG、GraphRAG、Modular RAG 等核心论文的深度解读，以及从 Naive RAG 到 Agentic RAG 的完整演进脉络。
>
> 💡 **前沿趋势：Agentic RAG**：2025 年以来，RAG 正在从静态的"检索-生成"管道演进为动态的 **Agentic RAG** 范式 [2]——Agent 不仅检索文档，还能判断何时需要检索、检索什么、对结果不满意时自动重写查询或换一个数据源。这本质上是将 RAG 从一个"流水线"升级为一个"会思考的检索 Agent"。

---

## 参考文献

[1] LEWIS P, PEREZ E, PIKTUS A, et al. Retrieval-augmented generation for knowledge-intensive NLP tasks[C]//NeurIPS. 2020.

[2] ASAI A, WU Z, WANG Y, et al. Self-RAG: Learning to retrieve, generate, and critique through self-reflection[C]//ICLR. 2024.

[3] GUAN X, LIU Y, LIN H, et al. CRAG — Comprehensive RAG benchmark[R]. arXiv preprint arXiv:2406.04744, 2024.

---

*下一节：[7.2 文档加载与文本分割](./02_document_loading.md)*
