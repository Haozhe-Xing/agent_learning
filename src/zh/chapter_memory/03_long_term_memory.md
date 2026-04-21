# 5.3 长期记忆：向量数据库与检索

长期记忆让 Agent 能够"记住"跨越多次会话的信息——你今天告诉它你是 Python 开发者、偏好简洁风格，下周再来对话时它依然记得。

> 📄 **学术前沿**：记忆系统的一个经典研究是 Stanford 的 Generative Agents [1]——25 个 AI Agent 在虚拟小镇中生活，它们通过"记忆流"（Memory Stream）记录所有经历，并通过**时近性（Recency）、重要性（Importance）、相关性（Relevance）** 三个维度的加权来检索记忆。特别值得注意的是其记忆衰减机制：越久远的记忆检索权重越低（指数衰减），但如果某段记忆被频繁回忆则权重会提升。这种设计与认知科学中的记忆巩固理论高度一致。

短期记忆（对话历史）在会话结束后就消失了。而长期记忆需要**持久化存储**——信息必须写入某种数据库，下次启动时能够检索回来。但传统数据库的精确查询（SQL `WHERE` 子句）并不适合自然语言场景。用户可能问"我之前提到过什么编程语言？"，你不可能预先知道该搜"Python"还是"Java"还是"Rust"。

这就是**向量数据库**发挥作用的地方。它的核心思想是：将文本转化为数学向量（一组浮点数），然后通过计算向量之间的距离来衡量语义相似度。"Python 是编程语言"和"编程用的 Python"虽然文字不同，但它们的向量非常接近。这让我们能够用自然语言作为查询条件，按语义相关性检索信息。

## 向量数据库的工作原理

![向量数据库工作原理](../svg/chapter_memory_03_vector_flow.svg)

核心思想：**语义相似的文本，会有相近的向量表示**。

下面的代码演示了这一点——我们生成三段文本的向量嵌入，然后计算它们之间的余弦相似度。你会看到，语义相似的句子（即使用词不同）相似度接近 1.0，而语义无关的句子相似度很低。

```python
import chromadb
from openai import OpenAI
import json
import datetime
from typing import Optional

client = OpenAI()

# ============================
# 向量嵌入工具
# ============================

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """获取文本的向量嵌入"""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# 测试：语义相似的文本有相近的向量
from numpy import dot
from numpy.linalg import norm

def cosine_similarity(v1, v2) -> float:
    """计算余弦相似度"""
    return dot(v1, v2) / (norm(v1) * norm(v2))

# 验证语义相似性
texts = [
    "Python 是一种编程语言",  # 原始句
    "Python 是用于编程的语言",  # 语义相似
    "今天天气很好",            # 语义不相关
]

embeddings = [get_embedding(t) for t in texts]
sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])
print(f"相似度（语义相似）：{sim_1_2:.4f}")  # 应该 > 0.9
print(f"相似度（语义不同）：{sim_1_3:.4f}")  # 应该 < 0.5
```

## 使用 ChromaDB 构建记忆系统

有了向量嵌入的概念基础，我们可以构建一个完整的长期记忆系统了。[ChromaDB](https://www.trychroma.com/) 是一个轻量级的开源向量数据库，支持持久化存储和语义检索，非常适合本地开发和原型构建。

下面的 `LongTermMemory` 类封装了记忆系统的核心操作：**添加记忆**（将文本向量化后存入数据库）、**语义检索**（根据查询找到最相关的记忆）、**分类管理**（区分偏好、事实、任务等不同类型的记忆）。

设计上有几个值得注意的点：
- **每个用户独立的 Collection**：避免不同用户的记忆互相干扰
- **重要性评分**：每条记忆都有 1-10 的重要性分数，检索时可以过滤低重要性的记忆
- **相关性阈值**：`format_for_prompt` 方法将检索到的记忆标记为"高度相关"、"相关"和"参考"，帮助 LLM 判断哪些记忆更值得参考

```python
class LongTermMemory:
    """
    基于 ChromaDB 的长期记忆系统
    支持：添加记忆、语义检索、更新、删除
    """
    
    def __init__(self, user_id: str, persist_dir: str = "./memory_db"):
        self.user_id = user_id
        
        # 创建持久化向量数据库
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # 每个用户有独立的 Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=f"user_{user_id}_memory",
            metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
        )
        
        print(f"[记忆系统] 加载用户 {user_id} 的记忆库，"
              f"已有 {self.collection.count()} 条记忆")
    
    def _embed(self, text: str) -> list[float]:
        """生成文本嵌入"""
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "general",
        importance: int = 5,
        source: str = "conversation"
    ) -> str:
        """
        添加一条记忆
        
        Args:
            content: 记忆内容
            memory_type: 类型（preference/fact/event/task）
            importance: 重要性（1-10）
            source: 来源
        
        Returns:
            记忆ID
        """
        import uuid
        
        memory_id = str(uuid.uuid4())
        
        self.collection.add(
            ids=[memory_id],
            embeddings=[self._embed(content)],
            documents=[content],
            metadatas=[{
                "type": memory_type,
                "importance": importance,
                "source": source,
                "created_at": datetime.datetime.now().isoformat(),
                "user_id": self.user_id
            }]
        )
        
        print(f"[记忆] 已保存：{content[:50]}...")
        return memory_id
    
    def search_memories(
        self,
        query: str,
        n_results: int = 5,
        memory_type: Optional[str] = None,
        min_importance: int = 1
    ) -> list[dict]:
        """
        语义搜索记忆
        
        Args:
            query: 查询内容（自然语言）
            n_results: 返回结果数量
            memory_type: 过滤类型
            min_importance: 最低重要性阈值
        
        Returns:
            相关记忆列表，按相似度排序
        """
        query_embedding = self._embed(query)
        
        # 构建过滤条件
        where = {"user_id": self.user_id}
        if memory_type:
            where["type"] = memory_type
        if min_importance > 1:
            where["importance"] = {"$gte": min_importance}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            # 如果没有足够的结果
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
        
        memories = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )):
                memories.append({
                    "content": doc,
                    "type": meta.get("type"),
                    "importance": meta.get("importance"),
                    "created_at": meta.get("created_at"),
                    "relevance": 1 - dist  # 转换为相似度（0-1）
                })
        
        return memories
    
    def get_all_memories(self, memory_type: Optional[str] = None) -> list[dict]:
        """获取所有记忆"""
        where = {"user_id": self.user_id}
        if memory_type:
            where["type"] = memory_type
        
        if self.collection.count() == 0:
            return []
        
        results = self.collection.get(
            where=where,
            include=["documents", "metadatas"]
        )
        
        memories = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            memories.append({
                "content": doc,
                "type": meta.get("type"),
                "importance": meta.get("importance"),
                "created_at": meta.get("created_at"),
            })
        
        # 按重要性排序
        return sorted(memories, key=lambda x: x.get("importance", 0), reverse=True)
    
    def format_for_prompt(self, memories: list[dict]) -> str:
        """将记忆格式化为 Prompt 可用的文本"""
        if not memories:
            return "无相关记忆"
        
        lines = ["【相关记忆】"]
        for m in memories:
            relevance = m.get("relevance", 0)
            if relevance >= 0.7:
                relevance_label = "高度相关"
            elif relevance >= 0.5:
                relevance_label = "相关"
            else:
                relevance_label = "参考"
            
            lines.append(f"[{m['type']} | {relevance_label}] {m['content']}")
        
        return "\n".join(lines)


接下来是记忆系统中最"智能"的部分——**自动记忆提取器**。与其让用户手动告诉 Agent"请记住这个信息"，不如让系统在每轮对话结束后自动分析：这轮对话中有没有值得长期记忆的信息？

`MemoryExtractor` 利用 LLM 来完成这个判断。它分析每轮对话的用户消息和 Agent 回复，提取出用户的个人信息、偏好、正在做的项目等有持久价值的内容，忽略闲聊和临时性查询。提取出的记忆会带有类型标签和重要性评分，方便后续的分类检索。

class MemoryExtractor:
    """从对话中自动提取值得记忆的信息"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def extract_memories(self, user_message: str, agent_reply: str) -> list[dict]:
        """
        分析对话，提取值得记忆的信息
        
        Returns:
            记忆列表，每条包含 content、type、importance
        """
        prompt = f"""分析以下对话，提取值得长期记忆的重要信息。

用户说：{user_message}
助手回复：{agent_reply}

提取规则：
- 记录用户的个人信息、偏好、习惯
- 记录重要的决策和结论
- 记录正在进行的项目或任务
- 忽略闲聊、日常问候、重复信息
- 忽略临时性的查询（如"今天几号"）

返回 JSON 格式（若无值得记忆的内容则返回空列表）：
[
  {{
    "content": "记忆内容（简洁陈述句）",
    "type": "preference|fact|event|task|skill",
    "importance": 1-10的整数
  }}
]

只返回 JSON，不要其他内容。"""
        
        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            # 处理可能的嵌套格式
            if isinstance(result, dict):
                memories = result.get("memories", result.get("items", []))
            else:
                memories = result
            return memories if isinstance(memories, list) else []
        except:
            return []


# ============================
# 测试
# ============================

def test_memory_system():
    """测试记忆系统"""
    
    # 初始化
    memory = LongTermMemory(user_id="user_001")
    extractor = MemoryExtractor()
    
    # 添加一些记忆
    memory.add_memory("用户叫张伟，是一名 Python 后端工程师", "fact", importance=8)
    memory.add_memory("用户偏好简洁的代码风格，不喜欢过度注释", "preference", importance=7)
    memory.add_memory("用户正在开发一个基于 FastAPI 的微服务项目", "task", importance=8)
    memory.add_memory("用户擅长 Python 和 Go 语言", "skill", importance=6)
    
    print("\n=== 语义搜索测试 ===")
    
    # 测试1：搜索代码风格相关
    results = memory.search_memories("写代码的时候有什么注意事项？")
    print("\n查询：写代码的时候有什么注意事项？")
    for r in results[:3]:
        print(f"  [{r['relevance']:.2f}] {r['content']}")
    
    # 测试2：搜索项目相关
    results = memory.search_memories("用户在做什么项目？")
    print("\n查询：用户在做什么项目？")
    for r in results[:3]:
        print(f"  [{r['relevance']:.2f}] {r['content']}")
    
    print("\n=== 自动提取记忆测试 ===")
    
    # 测试从对话中提取记忆
    new_memories = extractor.extract_memories(
        user_message="我最近开始学习 Rust，觉得所有权系统很有意思",
        agent_reply="Rust 的所有权系统确实很独特，是保证内存安全的核心机制..."
    )
    
    print("从对话中提取的记忆：")
    for m in new_memories:
        print(f"  [{m['type']} | 重要性:{m['importance']}] {m['content']}")
        memory.add_memory(m['content'], m['type'], m['importance'])
    
    return memory

# memory = test_memory_system()
```

---

## 小结

长期记忆的核心要素：
- **向量嵌入**：将文本转为可比较的向量
- **语义检索**：找到语义相关（而非关键词匹配）的记忆
- **分类存储**：区分偏好、事实、任务等不同类型
- **自动提取**：从对话中自动识别值得记忆的信息

---

*下一节：[5.4 工作记忆：Scratchpad 模式](./04_working_memory.md)*

---

## 参考文献

[1] PARK J S, O'BRIEN J C, CAI C J, et al. Generative agents: Interactive simulacra of human behavior[C]//UIST. 2023.

[2] JOHNSON J, DOUZE M, JÉGOU H. Billion-scale similarity search with GPUs[J]. IEEE Transactions on Big Data, 2021, 7(3): 535-547.
