# 6.7 进阶 RAG：GraphRAG 与 Agentic RAG 工程实战

> **本节目标**：超越"检索→生成"的朴素管道，掌握两种 2025 年生产级 RAG 架构——知识图谱增强检索（GraphRAG）和 Agent 主导的智能检索（Agentic RAG）——并能在真实项目中落地。

---

## 为什么需要进阶 RAG？

朴素 RAG（Naive RAG）的核心问题可以用一句话概括：**它把所有问题都当成局部问题来回答**。

```
朴素 RAG 的隐含假设：
  "用户想知道的一定在某几个相邻的文本块里"

现实中的反例：
  Q1: "这份报告中，所有部门的工作交叉点在哪里？"  → 需要全局视野
  Q2: "A公司 和 B公司 有什么间接合作关系？"       → 需要关系推理
  Q3: "为什么最终结论是X？请逐步推导"             → 需要多跳检索
  Q4: "用户手册里有没有前后矛盾的地方？"           → 需要对比多处文档
```

两种进阶架构分别针对这两类问题：

| 问题类型 | 适合架构 | 核心思想 |
|---------|---------|---------|
| 全局关系 / 跨文档推理 | **GraphRAG** | 把知识图谱化，检索走图结构 |
| 多跳 / 自适应 / 不确定性问题 | **Agentic RAG** | Agent 动态决策检索策略 |

---

## 第一部分：GraphRAG 工程实战

### 1.1 核心思想：从文本块到知识图谱

GraphRAG 的关键洞察：**文本块（Chunk）保留了知识，但丢失了关系**。

```
传统向量化：
  "苹果公司收购了 Shazam"  →  [0.23, -0.11, 0.87, ...]
  "Shazam 被谷歌的竞争对手买走了"  →  [0.21, -0.09, 0.85, ...]
  （两段向量很近，但你不知道苹果=谷歌的竞争对手这一推断链）

GraphRAG 的图化：
  节点：苹果公司、Shazam、谷歌
  边：苹果公司 --[收购]--> Shazam
      苹果公司 --[竞争对手]--> 谷歌
  （关系显式存储，支持图遍历推断）
```

### 1.2 两种检索模式

GraphRAG 提供 Local 和 Global 两种检索模式，适用场景完全不同：

```python
"""
GraphRAG 双模式检索示意

Local Search  → 适合具体问题
  "张三在项目中负责什么？"
  流程：找到"张三"节点 → 遍历邻居关系 → 拼接相关文本

Global Search → 适合全局问题
  "整个项目中，谁是最核心的协作节点？"
  流程：对所有社区摘要 Map-Reduce → 综合分析
"""
```

### 1.3 用 Microsoft GraphRAG 库快速搭建

微软开源了官方实现，以下是从零到可用的完整步骤：

```bash
# 安装
pip install graphrag

# 初始化工作空间
mkdir my_graphrag_project && cd my_graphrag_project
python -m graphrag init --root .

# 目录结构
# my_graphrag_project/
# ├── input/          ← 放你的 .txt 文档
# ├── settings.yaml   ← 配置文件（模型、嵌入等）
# └── output/         ← 索引输出
```

```yaml
# settings.yaml 关键配置（精简版）
llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4.1-mini         # 索引阶段用 mini 节省成本
  max_tokens: 4000

embeddings:
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

chunks:
  size: 1200
  overlap: 100

entity_extraction:
  max_gleanings: 1            # LLM 提取实体的重试次数，1 即可

community_reports:
  max_length: 2000            # 每个社区摘要的最大长度
```

```bash
# 构建索引（会调用 LLM 提取实体和关系，耗时 10-30 分钟）
python -m graphrag index --root .

# 成本估算（1000 篇短文档，约 500K tokens）
# gpt-4.1-mini: ~$0.4-1 
# text-embedding-3-small: ~$0.02
```

```python
# 查询（Python API）
import asyncio
from graphrag.query.cli import run_local_search, run_global_search

async def query_graphrag(question: str, mode: str = "local"):
    """
    mode: "local"  → 具体问题，基于实体邻居
          "global" → 全局问题，基于社区摘要
    """
    if mode == "local":
        result = await run_local_search(
            root_dir=".",
            query=question,
        )
    else:
        result = await run_global_search(
            root_dir=".",
            query=question,
        )
    return result

# 使用示例
async def main():
    # 局部问题 → local 模式
    answer1 = await query_graphrag(
        "GPT-4 的 Vision 功能是什么时候发布的？",
        mode="local"
    )
    
    # 全局问题 → global 模式
    answer2 = await query_graphrag(
        "这批文档中，哪些技术方向是被反复提及的核心主题？",
        mode="global"
    )
    
    print("局部查询：", answer1)
    print("全局查询：", answer2)

asyncio.run(main())
```

### 1.4 用 LightRAG 替代方案（成本更低）

如果 GraphRAG 的索引成本让你望而却步，LightRAG 是更实用的选择：

```bash
pip install lightrag-hku
```

```python
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embedding

async def build_lightrag_index(documents: list[str]) -> LightRAG:
    """构建 LightRAG 索引"""
    rag = LightRAG(
        working_dir="./lightrag_data",
        llm_model_func=gpt_4o_mini_complete,       # 用 mini 降低成本
        embedding_func=openai_embedding,
    )
    
    # 批量插入文档
    for doc in documents:
        await rag.ainsert(doc)
    
    return rag

async def query_lightrag(rag: LightRAG, question: str, mode: str = "hybrid"):
    """
    mode 选项：
      "naive"   → 传统向量检索（对比基线）
      "local"   → 基于实体邻居的局部检索
      "global"  → 基于高层概念的全局检索
      "hybrid"  → local + global 混合（推荐）
    """
    result = await rag.aquery(
        question,
        param=QueryParam(mode=mode)
    )
    return result

# 示例：增量插入新文档（LightRAG 的核心优势）
async def incremental_update(rag: LightRAG, new_doc: str):
    """新增文档无需重建整个图"""
    await rag.ainsert(new_doc)
    print("文档已增量添加到知识图谱")
```

### 1.5 GraphRAG vs 传统 RAG：何时选哪个？

```python
def choose_rag_strategy(use_case: dict) -> str:
    """
    根据使用场景选择 RAG 策略
    
    Returns: "naive_rag" | "graphrag" | "lightrag" | "hybrid"
    """
    
    # 场景特征
    has_global_questions = use_case.get("global_questions", False)
    knowledge_base_size = use_case.get("kb_size_docs", 100)
    budget_sensitive = use_case.get("budget_sensitive", False)
    frequent_updates = use_case.get("frequent_updates", False)
    
    # 决策逻辑
    if not has_global_questions and knowledge_base_size < 500:
        return "naive_rag"          # 够用，不要过度工程化
    
    if has_global_questions:
        if budget_sensitive or frequent_updates:
            return "lightrag"       # 图增强 + 低成本 + 增量更新
        else:
            return "graphrag"       # 微软官方，质量最高
    
    # 混合策略：本地用向量，跨文档用图
    return "hybrid"

# 典型使用场景对照
scenarios = {
    "FAQ 问答机器人（< 200 篇文档）":
        {"global_questions": False, "kb_size_docs": 200},
    
    "企业知识库助手（> 5000 篇，偶有全局问题）":
        {"global_questions": True, "kb_size_docs": 5000, "budget_sensitive": True},
    
    "学术论文分析（静态语料，需要关系推理）":
        {"global_questions": True, "kb_size_docs": 1000, "budget_sensitive": False},
    
    "新闻监控系统（每日更新）":
        {"global_questions": True, "frequent_updates": True, "budget_sensitive": True},
}

for scenario, config in scenarios.items():
    choice = choose_rag_strategy(config)
    print(f"{scenario}\n  → 推荐：{choice}\n")
```

---

## 第二部分：Agentic RAG 工程实战

### 2.1 核心思想：让 Agent 掌控检索决策

朴素 RAG 是一条固定的水管：问题进，答案出。Agentic RAG 是一个会思考的侦探：

```
朴素 RAG 水管：
  问题 → [检索Top-K] → [拼接Prompt] → [生成] → 答案

Agentic RAG 侦探：
  问题
    ↓
  「我需要查文档吗？还是我已经知道答案？」
    ↓（需要查）
  「用什么关键词查？原始问题够好还是需要改写？」
    ↓
  检索 → 「查到的东西够回答这个问题吗？」
    ↓（不够）
  「再换个角度查，或者换个数据源」
    ↓（够了）
  「我现在需要更多背景信息来支撑这个回答吗？」
    ↓
  生成答案 → 「这个答案有事实支撑吗？能追溯来源吗？」
    ↓
  最终答案（带引用）
```

### 2.2 四大核心组件

```python
from openai import OpenAI
from typing import Optional, Literal
import json

client = OpenAI()

# ── 组件 1：检索决策器 ──────────────────────────────────────────────
def should_retrieve(question: str, chat_history: list[dict]) -> bool:
    """判断当前问题是否需要检索"""
    
    prompt = f"""你是一个检索决策专家。判断以下问题是否需要查阅外部文档。

以下情况【不需要】检索：
- 简单计算或逻辑推理
- 通用常识（人人皆知的事实）
- 问题已在对话历史中被完整回答过

以下情况【需要】检索：
- 涉及特定领域、公司内部或时效性信息
- 需要精确数据、引用来源
- 问题较复杂，可能需要参考文档

对话历史：{json.dumps(chat_history[-3:], ensure_ascii=False)}
当前问题：{question}

只回复 "YES" 或 "NO"。"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    return response.choices[0].message.content.strip().upper() == "YES"


# ── 组件 2：查询改写器 ──────────────────────────────────────────────
def rewrite_query(original_question: str, context: str = "") -> list[str]:
    """将用户问题改写为更适合检索的查询（可产出多个变体）"""
    
    prompt = f"""将以下用户问题改写为 2-3 个检索查询变体。
要求：
1. 去掉口语化表达，换成更正式的关键词组合
2. 覆盖问题的不同侧面
3. 每个变体一行，不要编号或符号

背景信息（如有）：{context}
用户问题：{original_question}"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.3,
    )
    
    queries = response.choices[0].message.content.strip().split("\n")
    return [q.strip() for q in queries if q.strip()]


# ── 组件 3：检索质量评估器 ──────────────────────────────────────────
def evaluate_retrieval(
    question: str,
    retrieved_docs: list[str],
) -> dict:
    """评估检索结果是否足以回答问题"""
    
    docs_text = "\n---\n".join(retrieved_docs[:5])
    
    prompt = f"""评估以下检索结果是否足以回答用户问题。

用户问题：{question}

检索到的文档：
{docs_text}

请评估：
1. relevance: 文档与问题的相关程度（0-10）
2. sufficiency: 这些文档是否足以回答问题（true/false）
3. missing: 如果不足够，还缺少什么信息

以 JSON 格式回复。"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=200,
    )
    return json.loads(response.choices[0].message.content)


# ── 组件 4：答案生成器（带引用） ────────────────────────────────────
def generate_with_citation(
    question: str,
    docs: list[dict],  # [{"content": "...", "source": "doc_name", "page": 3}]
) -> dict:
    """基于检索文档生成答案，附带引用来源"""
    
    docs_formatted = ""
    for i, doc in enumerate(docs, 1):
        docs_formatted += f"\n[{i}] 来源：{doc['source']} 第{doc.get('page', '?')}页\n{doc['content']}\n"
    
    prompt = f"""基于以下参考文档回答用户问题。
要求：
- 如引用文档内容，用 [数字] 标注来源
- 如文档不足以回答，明确说明
- 不要凭空添加文档中没有的信息

参考文档：{docs_formatted}

用户问题：{question}"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": [f"{d['source']} p.{d.get('page', '?')}" for d in docs],
    }
```

### 2.3 用 LangGraph 编排完整 Agentic RAG 流程

LangGraph 的状态机非常适合编排 Agentic RAG 的动态决策流程：

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator


# ── 状态定义 ────────────────────────────────────────────────────────
class AgenticRAGState(TypedDict):
    question: str
    chat_history: list[dict]
    
    # 检索相关
    needs_retrieval: bool
    rewritten_queries: list[str]
    retrieved_docs: list[dict]
    retrieval_quality: dict
    retry_count: int
    
    # 输出
    final_answer: str
    sources: list[str]


# ── 节点函数 ────────────────────────────────────────────────────────
def decide_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """节点 1：判断是否需要检索"""
    needs = should_retrieve(
        state["question"],
        state.get("chat_history", [])
    )
    return {**state, "needs_retrieval": needs}


def rewrite_queries(state: AgenticRAGState) -> AgenticRAGState:
    """节点 2：改写查询"""
    queries = rewrite_query(state["question"])
    return {**state, "rewritten_queries": queries}


def retrieve_documents(state: AgenticRAGState) -> AgenticRAGState:
    """节点 3：执行检索（对每个改写查询检索并合并去重）"""
    all_docs = []
    seen_ids = set()
    
    for query in state.get("rewritten_queries", [state["question"]]):
        # 这里替换为你的实际向量数据库检索
        docs = mock_vector_search(query, top_k=3)
        for doc in docs:
            if doc["id"] not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc["id"])
    
    return {**state, "retrieved_docs": all_docs}


def assess_retrieval(state: AgenticRAGState) -> AgenticRAGState:
    """节点 4：评估检索质量"""
    doc_texts = [d["content"] for d in state["retrieved_docs"]]
    quality = evaluate_retrieval(state["question"], doc_texts)
    return {
        **state,
        "retrieval_quality": quality,
        "retry_count": state.get("retry_count", 0),
    }


def generate_answer(state: AgenticRAGState) -> AgenticRAGState:
    """节点 5：生成最终答案"""
    result = generate_with_citation(
        state["question"],
        state["retrieved_docs"]
    )
    return {
        **state,
        "final_answer": result["answer"],
        "sources": result["sources"],
    }


def answer_directly(state: AgenticRAGState) -> AgenticRAGState:
    """节点 6：无需检索，直接回答"""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            *state.get("chat_history", []),
            {"role": "user", "content": state["question"]}
        ],
    )
    return {
        **state,
        "final_answer": response.choices[0].message.content,
        "sources": [],
    }


# ── 条件路由函数 ─────────────────────────────────────────────────────
def route_after_decision(state: AgenticRAGState) -> str:
    """根据是否需要检索决定下一步"""
    return "rewrite_queries" if state["needs_retrieval"] else "answer_directly"


def route_after_assessment(state: AgenticRAGState) -> str:
    """根据检索质量决定：生成答案 or 重试"""
    quality = state.get("retrieval_quality", {})
    retry_count = state.get("retry_count", 0)
    
    # 质量足够 or 已重试 2 次 → 生成答案
    if quality.get("sufficiency", False) or retry_count >= 2:
        return "generate_answer"
    
    # 质量不足 → 重试（换查询策略）
    return "rewrite_queries"


# ── 构建图 ───────────────────────────────────────────────────────────
def build_agentic_rag_graph() -> StateGraph:
    graph = StateGraph(AgenticRAGState)
    
    # 添加节点
    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("rewrite_queries", rewrite_queries)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("assess_retrieval", assess_retrieval)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("answer_directly", answer_directly)
    
    # 设置入口
    graph.set_entry_point("decide_retrieval")
    
    # 条件边
    graph.add_conditional_edges(
        "decide_retrieval",
        route_after_decision,
        {
            "rewrite_queries": "rewrite_queries",
            "answer_directly": "answer_directly",
        }
    )
    
    graph.add_conditional_edges(
        "assess_retrieval",
        route_after_assessment,
        {
            "generate_answer": "generate_answer",
            "rewrite_queries": "rewrite_queries",    # 重试循环
        }
    )
    
    # 固定边
    graph.add_edge("rewrite_queries", "retrieve_documents")
    graph.add_edge("retrieve_documents", "assess_retrieval")
    graph.add_edge("generate_answer", END)
    graph.add_edge("answer_directly", END)
    
    return graph.compile()


# ── 使用示例 ─────────────────────────────────────────────────────────
agentic_rag = build_agentic_rag_graph()

def ask(question: str, history: list[dict] = []) -> dict:
    """对话接口"""
    result = agentic_rag.invoke({
        "question": question,
        "chat_history": history,
        "retry_count": 0,
    })
    return {
        "answer": result["final_answer"],
        "sources": result.get("sources", []),
        "used_retrieval": result.get("needs_retrieval", False),
    }

# 测试
response = ask("我们公司的产品退款政策是什么？")
print(f"答案：{response['answer']}")
print(f"来源：{response['sources']}")
print(f"是否检索：{response['used_retrieval']}")
```

### 2.4 生产环境的关键配置

```python
# ── 重试时的查询增强策略 ────────────────────────────────────────────
def enhance_query_after_failure(
    original_question: str,
    failed_queries: list[str],
    missing_info: str,
) -> list[str]:
    """检索失败后，生成更有针对性的查询"""
    
    prompt = f"""上一次检索没有找到足够的信息。
原始问题：{original_question}
已尝试的查询：{failed_queries}
缺失的信息：{missing_info}

请生成 2 个不同方向的新查询，避免重复已有查询的思路。
每行一个查询。"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]


# ── 多数据源路由 ─────────────────────────────────────────────────────
class MultiSourceRetriever:
    """在多个数据源之间动态路由检索"""
    
    def __init__(self):
        self.sources = {
            "internal_docs": self._search_internal,     # 内部文档向量库
            "web_search": self._search_web,             # 实时网络搜索
            "graph_rag": self._search_graph,            # 知识图谱检索
            "sql_database": self._search_sql,           # 结构化数据库
        }
    
    def route(self, question: str) -> list[str]:
        """决定使用哪些数据源"""
        prompt = f"""对于问题："{question}"
判断应该查询哪些数据源（可多选）：
- internal_docs: 公司内部文档
- web_search: 需要最新网络信息
- graph_rag: 需要实体关系推理
- sql_database: 需要精确数字或统计

只返回逗号分隔的数据源名称列表。"""
        
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0,
        )
        return [s.strip() for s in response.choices[0].message.content.split(",")]
    
    def search(self, question: str, queries: list[str]) -> list[dict]:
        """执行多源检索并合并结果"""
        sources_to_use = self.route(question)
        all_results = []
        
        for source_name in sources_to_use:
            if source_name in self.sources:
                results = self.sources[source_name](queries)
                for r in results:
                    r["source_type"] = source_name
                all_results.extend(results)
        
        return all_results
    
    def _search_internal(self, queries): ...   # 实现略
    def _search_web(self, queries): ...        # 实现略
    def _search_graph(self, queries): ...      # 实现略
    def _search_sql(self, queries): ...        # 实现略


# ── 流式输出支持 ─────────────────────────────────────────────────────
async def agentic_rag_stream(question: str):
    """支持流式输出的 Agentic RAG"""
    
    # 异步执行检索决策（这部分不流式）
    needs = should_retrieve(question, [])
    
    if needs:
        queries = rewrite_query(question)
        # 向量检索...
        docs = []  # 检索结果
        
        # 流式生成答案
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"基于以下文档回答：\n{docs}\n\n问题：{question}"
            }],
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    else:
        stream = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": question}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

## 第三部分：两种架构的对比与选型

### 3.1 核心差异对比

| 维度 | GraphRAG / LightRAG | Agentic RAG |
|------|---------------------|-------------|
| **适合问题类型** | 关系推理、全局归纳 | 多跳问答、不确定性问题 |
| **检索策略** | 图遍历 + 社区摘要 | 动态决策 + 多轮检索 |
| **索引成本** | 高（需预构建知识图谱） | 低（沿用普通向量索引） |
| **延迟** | 中（图遍历） | 高（多轮 LLM 调用） |
| **可解释性** | 强（图结构透明） | 中（决策链可追溯） |
| **维护难度** | 中（增量更新）| 高（状态机逻辑复杂）|
| **推荐场景** | 企业知识库、文档分析 | 客服、研究助手、复杂问答 |

### 3.2 组合使用：最强架构

生产环境中，两者往往结合使用：

```
用户提问
    ↓
Agentic RAG 决策层
    ├─ 简单具体问题 → 普通向量检索
    ├─ 关系推理问题 → LightRAG 图检索
    ├─ 全局分析问题 → GraphRAG Global Search
    └─ 实时信息需求 → Web 搜索
         ↓
    结果合并 + 质量评估
         ↓
    生成带引用的最终答案
```

```python
class HybridRAGSystem:
    """组合使用 Agentic RAG + GraphRAG 的完整系统"""
    
    def __init__(self, lightrag_dir: str, vector_store):
        self.lightrag = LightRAG(working_dir=lightrag_dir, ...)
        self.vector_store = vector_store
        self.multi_source = MultiSourceRetriever()
    
    async def query(self, question: str) -> dict:
        # 1. 分类问题类型
        question_type = self._classify_question(question)
        
        # 2. 路由到对应检索策略
        if question_type == "relational":
            docs = await self.lightrag.aquery(question, QueryParam(mode="hybrid"))
        elif question_type == "global":
            docs = await self.lightrag.aquery(question, QueryParam(mode="global"))
        else:
            # 普通向量检索 + Agentic 决策
            queries = rewrite_query(question)
            docs = self.vector_store.search(queries)
        
        # 3. 生成答案
        return generate_with_citation(question, docs)
    
    def _classify_question(self, question: str) -> str:
        """分类：relational / global / factual"""
        ...
```

---

## 常见错误与调试

```python
# ❌ 错误 1：GraphRAG 索引阶段忘记检查 Token 消耗
# 症状：跑了一半发现超出 API 额度
# 解决：先用小样本（10-20 篇）测试，确认质量和成本再全量

# ❌ 错误 2：Agentic RAG 检索循环不设上限
# 症状：某些问题触发无限重试，导致超时和费用激增
graph.add_conditional_edges(
    "assess_retrieval",
    route_after_assessment,
    {
        "generate_answer": "generate_answer",
        "rewrite_queries": "rewrite_queries",   # ← 必须在 route 函数中设置 retry_count 上限
    }
)
# ✅ 正确做法：在 route_after_assessment 中加 retry_count >= 2 的硬性截止

# ❌ 错误 3：LightRAG 查询模式选错
# 症状：回答质量差，但换成 "hybrid" 模式后立刻变好
# 解决：默认用 "hybrid"，只有在调试时才用 "naive" 对比

# ❌ 错误 4：检索评估器提示词太宽松
# 症状：所有检索都被评为"足够"，实际上答案质量很差
# 解决：在评估提示词中明确定义"足够"的标准（如：必须包含具体数字/日期/专有名词）
```

---

## 本节小结

| 技术 | 核心价值 | 生产就绪 |
|------|---------|---------|
| **GraphRAG** | 处理全局关系问题，准确率最高 | ✅（微软官方维护）|
| **LightRAG** | GraphRAG 的轻量替代，支持增量更新 | ✅（低成本生产可用）|
| **Agentic RAG** | 动态检索决策，适应复杂多变的问题 | ✅（需配合 LangGraph）|
| **三者组合** | 覆盖所有 RAG 场景 | ⚠️（复杂度高，按需组合）|

---

## 练习题

1. **实战题**：用 LightRAG 对你的一个本地文档集（至少 20 篇）构建知识图谱，分别用 `local`、`global`、`hybrid` 三种模式查询同一个问题，对比回答质量差异，写一段分析。

2. **设计题**：一家电商公司有三个知识库：产品说明书（500篇）、客服历史对话（10万条）、实时库存数据库。设计一个 Agentic RAG 系统，说明如何路由不同类型的用户问题。

3. **调试题**：以下 Agentic RAG 代码运行后，对"公司最新营收是多少？"这种问题永远不会触发检索，分析原因并修复：

```python
def should_retrieve(question: str, history: list) -> bool:
    prompt = f"这个问题需要查资料吗？只回复YES/NO：{question}"
    # （提示：问题在提示词的歧义性上）
    ...
```

4. **进阶题**：GraphRAG 在处理中文文档时常见一个问题：实体提取质量下降（如把"腾讯公司"和"腾讯"识别为两个不同实体）。设计一个后处理步骤解决实体合并问题。

---

*上一节：[6.6 论文解读：RAG 前沿进展](./06_paper_readings.md)*  
*返回：[第6章 检索增强生成](./README.md)*
