# 4.7 实战：MemGPT/Letta 记忆架构工程实践

> **本节目标**：基于 MemGPT 的核心思想，实现一个生产级分层记忆 Agent，并了解 Letta 框架的使用方式。

---

## 从论文到工程：MemGPT 的核心启示

4.6 节介绍了 MemGPT 论文的核心思想——将 LLM 的上下文类比为操作系统内存，通过分层存储和自我编辑突破上下文窗口限制。本节将其核心思想落地为可运行的代码。

MemGPT 的工程化版本 **Letta**（2025 年更名）提供了完整的 Agent 记忆管理框架，但理解底层原理对于定制化开发至关重要。

---

## 分层记忆架构实现

### 三层记忆的分工

MemGPT 把 LLM 上下文类比为操作系统内存，采用"内存分级"思路：

| 层 | 放什么 | 类比 OS | 关键特点 |
|---|---|---|---|
| **Core Memory（核心记忆）** | 用户姓名、长期偏好、关键事实、当前目标 | 常驻内存/寄存器 | 始终在 Prompt 里，是模型回答的"常识依据" |
| **Working Memory（工作记忆）** | 当前任务相关的近期上下文 | 内存（RAM） | 容量有限，只保留最近若干条 |
| **Archive Memory（归档记忆）** | 大量历史内容、长文档 | 硬盘 / 外部存储 | 不进 Prompt，**按需检索**才调出来 |

**为什么要分层**：上下文窗口是有限且昂贵的资源。如果把所有历史都塞进 Prompt，很快就会超出窗口、且充满噪声。分层的核心思想是——**把最重要、最常用的信息常驻（Core），把次要的暂存（Working），把海量但偶尔才用的信息放到外部、用检索按需调取（Archive）**。

### 核心实现

整个 Agent 的关键在 `chat` 方法：自动管理记忆 → 构建含记忆的 Prompt → 调用 LLM → 处理记忆工具调用。

```python
class LayeredMemoryAgent:
    """分层记忆 Agent：Core + Working + Archive 三层。"""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        # 始终在 Prompt 中：放用户画像和关键偏好
        self.core_memory = {
            "user_name": "", "preferences": [],
            "key_facts": [], "active_goals": []
        }
        # 当前任务相关的短期信息
        self.working_memory = []
        # 持久化存储，模拟向量数据库
        self.archive_memory = []
        # 对话历史
        self.conversation = []

    def chat(self, user_input: str) -> str:
        """主对话入口：5 步流水线。"""
        # 1. 自动记忆管理：检查是否需要更新记忆
        self._auto_manage_memory(user_input)
        # 2. 构建含记忆的 Prompt
        messages = self._build_messages(user_input)
        # 3. 调用 LLM
        response = client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=2000, tools=self._get_memory_tools()
        )
        # 4. 处理工具调用（记忆自我编辑）
        reply = self._process_response(response)
        # 5. 保存到对话历史
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def _build_messages(self, user_input: str) -> list[dict]:
        """System Prompt 中始终注入核心记忆 + 近期工作记忆。"""
        system = f"""你是具备分层记忆能力的 Agent。
## 核心记忆（始终记住）
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}
## 工作记忆（当前任务相关）
{json.dumps(self.working_memory[-5:], ensure_ascii=False, indent=2)}
## 记忆管理指令
- 核心记忆中的信息是你的"常识"，始终作为回答依据
- 问到归档内容时，用 search_archive 工具检索
- 需要记新信息时，用 update_core_memory 工具
- 当前对话产生需长期保存的内容时，用 archive_content 工具"""
        recent = self.conversation[-20:]  # 滑动窗口：最近 10 轮
        return [{"role": "system", "content": system}] + recent \
             + [{"role": "user", "content": user_input}]

    # 其余方法（_auto_manage_memory / _get_memory_tools /
    # _process_response / _search_archive）见仓库完整实现
```

> 📦 **完整代码（约 250 行）见仓库** `examples/chapter04/layered_memory_agent.py`，含自动记忆提取 prompt、3 个记忆管理工具的 schema、工具调用分发逻辑。

### 使用示例

```python
agent = LayeredMemoryAgent()

# 第一轮：用户介绍自己
agent.chat("你好！我叫小明，我是一名数据科学家，平时喜欢用 Python")

# 第二轮：用户提出偏好
agent.chat("我比较喜欢简洁的回答，不要太啰嗦")

# 第三轮：用户讨论工作
agent.chat("我正在做一个客户流失预测项目，使用的是 XGBoost")

# 第四轮：验证记忆是否保持
print(agent.chat("我之前说我在做什么项目来着？"))
# Agent 应该能从核心记忆中回忆起"客户流失预测项目"
```

四轮对话中，**核心记忆自动捕获**了用户姓名（"小明"）、身份（"数据科学家"）、偏好（"简洁回答"）、项目（"客户流失预测"）。第五轮即使 Agent 重启，只要核心记忆持久化，就能正确回答。

---

## Letta 框架快速上手

Letta（原 MemGPT）是论文作者创办的商业化框架，提供了完整的分层记忆管理：

```python
# pip install letta
from letta import create_client

letta_client = create_client()

# 创建一个带分层记忆的 Agent
agent = letta_client.create_agent(
    name="memory_assistant",
    memory_blocks=[
        {"label": "persona", "value": "你是一个有帮助的AI助手，善于记住用户信息。"},
        {"label": "human",   "value": "用户信息待填写"},   # Agent 会自动更新
    ],
    llm="gpt-4.1",
    embedding="text-embedding-3-small",
)

# 与 Agent 对话
response = letta_client.send_message(
    agent_id=agent.id,
    message="你好！我叫小红，我在做 NLP 研究",
    role="user"
)
# Agent 会自动将"小红""NLP 研究"更新到 human 记忆块中
```

> 💡 **何时用 Letta 而不是自己实现**：
> - 多用户管理、记忆持久化、审计日志、模型路由——这些"非核心"能力 Letta 已经做好了
> - 如果你的项目里"分层记忆"是核心创新点，自己实现更可控
> - 简单原型或学习目的，推荐 Letta（生产级稳定性）

---

## 记忆衰减与遗忘工程

人脑不是什么都记——重要的记住，不重要的逐渐遗忘。Agent 的记忆也应如此：

```python
import math, time

# 记忆类型及其衰减速率：身份信息永不衰减，琐碎信息快速衰减
DECAY_RATES = {
    "identity": 0.0,    # 永不衰减
    "preference": 0.01, # 缓慢衰减
    "fact": 0.05,       # 中速衰减
    "context": 0.1,     # 快速衰减
    "trivial": 0.3,     # 极快衰减
}

class MemoryWithDecay:
    """带衰减机制 + 访问增强的记忆系统。"""

    def __init__(self):
        self.memories: list[dict] = []  # 每条含 content/type/importance/created_at/access_count

    def add(self, content: str, type: str, importance: float = 0.5):
        self.memories.append({
            "content": content, "type": type, "importance": importance,
            "created_at": time.time(), "access_count": 0
        })

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """综合考虑：相关性、时间衰减、访问增强。"""
        scored = []
        for mem in self.memories:
            relevance = self._compute_relevance(query, mem["content"])
            # 时间衰减：越久远的记忆强度越低
            age_hours = (time.time() - mem["created_at"]) / 3600
            decay = math.exp(-DECAY_RATES.get(mem["type"], 0.05) * age_hours)
            # 访问增强：经常被检索的记忆不容易遗忘
            access_bonus = min(0.2, mem["access_count"] * 0.02)
            # 综合评分
            score = relevance * 0.4 + mem["importance"] * decay * 0.4 + access_bonus * 0.2
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, mem in scored[:top_k]:
            mem["access_count"] += 1   # 访问计数+1（强化记忆）
            results.append({
                "content": mem["content"], "score": score,
                "type": mem["type"], "age_hours": (time.time() - mem["created_at"]) / 3600
            })
        return results

    def cleanup(self, threshold: float = 0.01) -> str:
        """清理衰减到阈值的记忆。"""
        before = len(self.memories)
        self.memories = [m for m in self.memories if self._current_strength(m) > threshold]
        return f"已清理 {before - len(self.memories)} 条衰减记忆，剩余 {len(self.memories)} 条"
```

### 三件事值得专门讲

1. **"全都记住"是错的**：存储成本是次要问题，**检索质量**才是核心。琐碎记忆挤占 top_k 名额，反而让重要信息被漏掉。
2. **衰减率分级体现信息价值差异**：身份信息几乎永远有用，琐碎信息很快失去价值——遗忘不是缺陷，是主动的信息筛选机制。
3. **访问增强补充单纯时间衰减**：经常被检索的记忆获得加成（"越回忆越牢固"），模拟人脑的记忆巩固。

---

## 小结

| 概念 | 说明 |
|---|---|
| 分层记忆 | Core + Working + Archive 三层，对应 OS 内存分级 |
| 记忆自管理 | Agent 通过工具调用主动管理自己的记忆（MemGPT 核心思想） |
| Letta 框架 | MemGPT 的商业化版本，提供完整的分层记忆管理 |
| 记忆衰减 | 不同类型记忆有不同衰减速率，重要记忆永不遗忘 |
| 访问增强 | 经常被检索的记忆不易遗忘（模拟人脑"回忆强化"） |

> 📖 **延伸阅读**：
> - Packer et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.
> - Letta Documentation. https://docs.letta.com, 2025.
> - Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST, 2023.

---

## 📝 本章练习

**练习 1（概念）**：MemGPT 把 LLM 的上下文类比为操作系统的内存。本节实现的分层记忆 Agent 有三层：Core Memory、Working Memory、Archive Memory。请用自己的话说清这三层各自放什么、为什么要这样分层，并解释它对应操作系统里的哪个概念。

<details>
<summary>参考答案</summary>

三层记忆的分工：

| 层 | 放什么 | 类比操作系统 | 关键特点 |
|---|---|---|---|
| **Core Memory（核心记忆）** | 用户姓名、长期偏好、关键事实、当前目标 | 常驻内存 / 寄存器 | **始终**在 Prompt 里，是模型回答的"常识依据" |
| **Working Memory（工作记忆）** | 当前任务相关的近期上下文 | 内存（RAM） | 容量有限，只保留最近若干条 |
| **Archive Memory（归档记忆）** | 大量历史内容、长文档 | 硬盘 / 外部存储 | 不进 Prompt，**按需检索**才调出来 |

**为什么要分层：** 上下文窗口是有限且昂贵的资源。如果把所有历史都塞进 Prompt，很快就会超出窗口、而且充满噪声。分层的核心思想是——**把最重要、最常用的信息常驻（Core），把次要的暂存（Working），把海量但偶尔才用的信息放到外部、用检索按需调取（Archive）**。这正是操作系统"内存分级（寄存器/RAM/硬盘）"的思路：用有限的快速存储装最热的数据，冷数据放慢速大容量存储。

这样既突破了上下文窗口的物理限制，又保证了关键信息（如"用户在做客户流失预测项目"）不会被遗忘。

</details>

**练习 2（辨析）**：本节的记忆衰减机制里，"identity"类记忆衰减率是 0.0（永不衰减），"trivial"类是 0.3（极快衰减）。有同学说："既然存储成本越来越低，干脆所有记忆都永不衰减，全都记住最保险。" 这个想法对吗？请结合检索质量分析。

<details>
<summary>参考答案</summary>

这个想法**不对**。问题不在存储成本，而在**检索质量和上下文噪声**。

- **"全都记住"会让检索被噪声淹没**：Agent 每次决策能放进上下文的记忆数量有限（top_k）。如果"用户今天喝了一杯咖啡"这种琐碎信息和"用户名叫小明""当前项目是客户流失预测"这种关键信息一起参与排序，琐碎记忆可能挤占宝贵的名额，反而让真正重要的信息被漏掉。
- **人脑也是这么做的**：本节开头就点明"人脑不是什么都记——重要的记住，不重要的逐渐遗忘"。遗忘不是缺陷，而是一种**主动的信息筛选机制**。
- **衰减率分级体现了信息的价值差异**：身份信息（identity）几乎永远有用，所以不衰减；琐碎信息（trivial）很快失去价值，让它快速衰减、被 `cleanup()` 清理掉，能保持记忆库的信噪比。
- 同时本节还设计了**访问增强**（access_count）：经常被检索的记忆会获得加成、不容易被遗忘——这模拟了人脑"越回忆越牢固"的特性，是对单纯时间衰减的补充。

所以正确思路是**有区别地遗忘**：重要的留住，琐碎的淘汰，被反复用到的强化。这样检索出来的才是真正相关、高质量的记忆。

</details>

**练习 3（动手）**：本节 `MemoryWithDecay` 的 `_compute_relevance` 用的是简单的"关键词重叠"来算相关性，对中文和近义词很不友好。请说明它的缺陷，并写出用**向量相似度（Embedding）** 改造后的思路与核心代码。

<details>
<summary>参考答案</summary>

**关键词重叠的缺陷：**
- 只能匹配**字面完全相同**的词，无法理解语义。查询"项目"匹配不到"客户流失预测工作"，因为没有共同的词。
- 对中文尤其不友好：中文不像英文用空格天然分词，`content.split()` 切不出合理的词。
- 无法处理近义词、上下位词（"手机"vs"智能手机"）。

**用 Embedding 改造的思路：** 把查询和记忆内容都编码成向量，用**余弦相似度**衡量语义接近程度——语义相近的向量在空间里距离也近，哪怕一个字都不重叠。

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class MemoryWithDecay:
    def __init__(self):
        self.memories = []
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def add(self, content, memory_type, importance=0.5):
        self.memories.append({
            "content": content, "type": memory_type, "importance": importance,
            "created_at": time.time(), "access_count": 0,
            "embedding": self.embedder.encode(content),   # 入库时缓存向量
        })

    def _compute_relevance(self, query: str, mem: dict) -> float:
        q_vec = self.embedder.encode(query)
        m_vec = mem["embedding"]
        cos = np.dot(q_vec, m_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(m_vec) + 1e-8)
        return (cos + 1) / 2   # 归一化到 [0, 1]
```

**讲解：**
- 入库（`add`）时就编码并缓存向量，检索时只需编码一次查询，避免重复开销。
- 余弦相似度衡量两个向量方向的接近程度，是语义检索的标准做法；这样"项目"就能和"客户流失预测工作"匹配上。
- 归一化到 [0,1] 是为了和 `retrieve` 里的 `importance × decay`、`access_bonus` 等分项在同一量纲上加权。
- 这其实就是把 4.3 节"长期记忆：向量数据库与检索"的思想，落到了记忆衰减系统里——真实生产中 Archive Memory 通常就用向量数据库（如 Milvus）实现。

</details>

---

[4.6 论文解读：记忆系统前沿进展](./06_paper_readings.md)
