# 4.7 实战：MemGPT/Letta 记忆架构工程实践

> **本节目标**：基于 MemGPT 的核心思想，实现一个生产级分层记忆 Agent，并了解 Letta 框架的使用方式。

---

## 从论文到工程：MemGPT 的核心启示

5.6 节介绍了 MemGPT 论文的核心思想——将 LLM 的上下文类比为操作系统内存，通过分层存储和自我编辑突破上下文窗口限制。本节将其核心思想落地为可运行的代码。

MemGPT 的工程化版本 **Letta**（2025 年更名）提供了完整的 Agent 记忆管理框架，但理解底层原理对于定制化开发至关重要。

---

## 分层记忆架构实现

### 核心设计

```python
from openai import OpenAI
import json
import time

client = OpenAI()


class LayeredMemoryAgent:
    """分层记忆 Agent（MemGPT 思想的工程实现）
    
    三层记忆架构：
    1. Core Memory（核心记忆）— 始终在上下文中，放最关键的信息
    2. Working Memory（工作记忆）— 当前任务相关的短期信息
    3. Archive Memory（归档记忆）— 外部存储，按需检索
    """
    
    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        
        # 核心记忆：始终在 Prompt 中，放用户画像和关键偏好
        self.core_memory = {
            "user_name": "",
            "preferences": [],
            "key_facts": [],
            "active_goals": [],
        }
        
        # 工作记忆：当前对话的相关上下文
        self.working_memory = []
        self.max_working_items = 10
        
        # 归档记忆：持久化存储，模拟向量数据库
        self.archive_memory = []
        
        # 对话历史
        self.conversation = []
    
    def chat(self, user_input: str) -> str:
        """主对话入口"""
        
        # 1. 自动记忆管理：检查是否需要更新记忆
        self._auto_manage_memory(user_input)
        
        # 2. 构建包含记忆的 Prompt
        messages = self._build_messages(user_input)
        
        # 3. 调用 LLM
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2000,
            tools=self._get_memory_tools()
        )
        
        # 4. 处理工具调用（记忆自我编辑）
        reply = self._process_response(response)
        
        # 5. 保存到对话历史
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": reply})
        
        return reply
    
    def _auto_manage_memory(self, user_input: str):
        """自动从用户输入中提取关键信息更新记忆"""
        
        # 使用小模型快速判断是否需要更新记忆
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"""分析以下用户输入，提取需要记住的关键信息。

当前核心记忆：
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}

用户输入：{user_input}

如果包含需要记忆的信息（如姓名、偏好、重要事实），返回 JSON：
{{"updates": {{"field": "值", ...}}, "archive": "需要归档的内容或null"}}

如果没有需要记忆的信息，返回：
{{"updates": {{}}, "archive": null}}"""
            }],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 更新核心记忆
        for key, value in result.get("updates", {}).items():
            if key in self.core_memory and value:
                if isinstance(self.core_memory[key], list):
                    if value not in self.core_memory[key]:
                        self.core_memory[key].append(value)
                else:
                    self.core_memory[key] = value
        
        # 归档长内容
        archive_content = result.get("archive")
        if archive_content:
            self.archive_memory.append({
                "content": archive_content,
                "timestamp": time.time(),
                "source": "auto_extract"
            })
    
    def _build_messages(self, user_input: str) -> list[dict]:
        """构建包含记忆的完整 Prompt"""
        
        # System Prompt：包含核心记忆
        system_prompt = f"""你是一个具备分层记忆能力的 Agent。

## 核心记忆（始终记住）
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}

## 工作记忆（当前任务相关）
{json.dumps(self.working_memory[-5:], ensure_ascii=False, indent=2)}

## 记忆管理指令
- 核心记忆中的信息是你的"常识"，始终作为回答的依据
- 如果用户问到归档记忆中的内容，使用 search_archive 工具搜索
- 如果需要记住新的重要信息，使用 update_core_memory 工具
- 如果当前对话产生了需要长期保存的内容，使用 archive_content 工具"""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        # 添加近期对话（保留最近 10 轮）
        recent = self.conversation[-20:]  # 10轮 = 20条消息
        messages.extend(recent)
        
        # 添加当前用户输入
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _get_memory_tools(self) -> list[dict]:
        """定义记忆管理工具"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_archive",
                    "description": "在归档记忆中搜索相关内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索关键词"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "update_core_memory",
                    "description": "更新核心记忆中的字段",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "field": {
                                "type": "string",
                                "description": "字段名（user_name/preferences/key_facts/active_goals）"
                            },
                            "value": {
                                "type": "string",
                                "description": "新的值"
                            }
                        },
                        "required": ["field", "value"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "archive_content",
                    "description": "将内容保存到归档记忆",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "要归档的内容"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
        ]
    
    def _process_response(self, response) -> str:
        """处理 LLM 响应，执行记忆工具调用"""
        message = response.choices[0].message
        
        # 如果没有工具调用，直接返回文本
        if not message.tool_calls:
            return message.content or ""
        
        # 执行工具调用
        tool_results = []
        text_parts = []
        
        if message.content:
            text_parts.append(message.content)
        
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if func_name == "search_archive":
                results = self._search_archive(args["query"])
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "result": json.dumps(results, ensure_ascii=False)
                })
                text_parts.append(f"[检索到归档内容: {len(results)} 条]")
            
            elif func_name == "update_core_memory":
                field = args["field"]
                value = args["value"]
                if field in self.core_memory:
                    if isinstance(self.core_memory[field], list):
                        self.core_memory[field].append(value)
                    else:
                        self.core_memory[field] = value
                    text_parts.append(f"[已更新核心记忆: {field}]")
            
            elif func_name == "archive_content":
                self.archive_memory.append({
                    "content": args["content"],
                    "timestamp": time.time(),
                    "source": "agent_archived"
                })
                text_parts.append("[已归档内容]")
        
        return "\n".join(text_parts)
    
    def _search_archive(self, query: str) -> list[dict]:
        """在归档记忆中搜索（简化版：关键词匹配）"""
        results = []
        query_lower = query.lower()
        
        for item in self.archive_memory:
            if query_lower in item["content"].lower():
                results.append(item)
        
        return results[:5]  # 最多返回5条


# 使用示例
agent = LayeredMemoryAgent()

# 第一轮：用户介绍自己
print(agent.chat("你好！我叫小明，我是一名数据科学家，平时喜欢用 Python"))

# 第二轮：用户提出偏好
print(agent.chat("我比较喜欢简洁的回答，不要太啰嗦"))

# 第三轮：用户讨论工作
print(agent.chat("我正在做一个客户流失预测项目，使用的是 XGBoost"))

# 第四轮：验证记忆是否保持
print(agent.chat("我之前说我在做什么项目来着？"))
# Agent 应该能从核心记忆中回忆起"客户流失预测项目"

# 查看核心记忆状态
print("\n当前核心记忆：")
print(json.dumps(agent.core_memory, ensure_ascii=False, indent=2))
```

---

## Letta 框架快速上手

Letta（原 MemGPT）是论文作者创办的商业化框架，提供了完整的分层记忆管理：

```python
# Letta 安装：pip install letta
# 详细文档：https://docs.letta.com

from letta import create_client

# 创建 Letta 客户端
letta_client = create_client()

# 创建一个带记忆的 Agent
agent = letta_client.create_agent(
    name="memory_assistant",
    memory_blocks=[
        {
            "label": "persona",
            "value": "你是一个有帮助的AI助手，善于记住用户信息。"
        },
        {
            "label": "human",
            "value": "用户信息待填写"  # Agent 会自动更新
        }
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

print(response.messages)

# Agent 会自动将用户信息更新到 human 记忆块中
# 后续对话中，Agent 能自动检索和回忆这些信息
```

---

## 记忆衰减与遗忘工程

人脑不是什么都记——重要的记住，不重要的逐渐遗忘。Agent 的记忆也应如此：

```python
import math
import time


class MemoryWithDecay:
    """带衰减机制的 Agent 记忆
    
    灵感来自 Generative Agents 论文的重要性评分 + 时间衰减
    """
    
    # 记忆类型及其衰减速率
    DECAY_RATES = {
        "identity": 0.0,      # 身份信息永不衰减
        "preference": 0.01,   # 偏好缓慢衰减
        "fact": 0.05,         # 事实性信息中速衰减
        "context": 0.1,       # 上下文信息快速衰减
        "trivial": 0.3,       # 琐碎信息极快衰减
    }
    
    def __init__(self):
        self.memories = []  # [{"content", "type", "importance", "created_at", "access_count"}]
    
    def add(self, content: str, memory_type: str, importance: float = 0.5):
        """添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型（决定衰减速率）
            importance: 重要性评分 0.0-1.0
        """
        self.memories.append({
            "content": content,
            "type": memory_type,
            "importance": importance,
            "created_at": time.time(),
            "access_count": 0,
        })
    
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """检索记忆，综合考虑相关性和衰减"""
        
        scored = []
        for mem in self.memories:
            # 1. 语义相关性（简化版：关键词匹配，实际应用中用 Embedding）
            relevance = self._compute_relevance(query, mem["content"])
            
            # 2. 时间衰减
            age_hours = (time.time() - mem["created_at"]) / 3600
            decay_rate = self.DECAY_RATES.get(mem["type"], 0.05)
            decay_factor = math.exp(-decay_rate * age_hours)
            
            # 3. 访问增强（经常被检索的记忆不容易遗忘）
            access_bonus = min(0.2, mem["access_count"] * 0.02)
            
            # 4. 综合评分
            score = (
                relevance * 0.4 +
                mem["importance"] * decay_factor * 0.4 +
                access_bonus * 0.2
            )
            
            scored.append((score, mem))
        
        # 排序并返回 top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, mem in scored[:top_k]:
            mem["access_count"] += 1  # 访问计数+1
            results.append({
                "content": mem["content"],
                "score": score,
                "type": mem["type"],
                "age_hours": (time.time() - mem["created_at"]) / 3600,
            })
        
        return results
    
    def cleanup(self, threshold: float = 0.01):
        """清理衰减到阈值的记忆"""
        before = len(self.memories)
        
        self.memories = [
            mem for mem in self.memories
            if self._current_strength(mem) > threshold
        ]
        
        removed = before - len(self.memories)
        return f"已清理 {removed} 条衰减记忆，剩余 {len(self.memories)} 条"
    
    def _compute_relevance(self, query: str, content: str) -> float:
        """计算语义相关性（简化版）"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words & content_words)
        return overlap / len(query_words)
    
    def _current_strength(self, mem: dict) -> float:
        """计算记忆当前强度"""
        age_hours = (time.time() - mem["created_at"]) / 3600
        decay_rate = self.DECAY_RATES.get(mem["type"], 0.05)
        decay_factor = math.exp(-decay_rate * age_hours)
        access_bonus = min(0.2, mem["access_count"] * 0.02)
        
        return mem["importance"] * decay_factor + access_bonus


# 使用示例
memory = MemoryWithDecay()

# 添加不同类型的记忆
memory.add("用户名叫小明", "identity", importance=1.0)
memory.add("用户偏好简洁的回答", "preference", importance=0.8)
memory.add("当前项目是客户流失预测", "fact", importance=0.6)
memory.add("上次对话提到了周末计划", "context", importance=0.3)
memory.add("用户今天喝了一杯咖啡", "trivial", importance=0.1)

# 检索记忆
results = memory.retrieve("项目")
for r in results:
    print(f"[{r['type']}] score={r['score']:.3f}: {r['content']}")

# 清理衰减记忆
print(memory.cleanup())
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 分层记忆 | Core Memory + Working Memory + Archive Memory 三层架构 |
| 记忆自管理 | Agent 通过工具调用主动管理自己的记忆（MemGPT 核心思想） |
| Letta 框架 | MemGPT 的商业化版本，提供完整的分层记忆管理 |
| 记忆衰减 | 不同类型的记忆有不同的衰减速率，重要的记忆永不遗忘 |
| 访问增强 | 经常被检索的记忆不容易遗忘（模拟人脑的"回忆强化"） |

> 📖 **延伸阅读**：
> - Packer et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.
> - Letta Documentation. https://docs.letta.com, 2025.
> - Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST, 2023.

---

[4.6 论文解读：记忆系统前沿进展](./06_paper_readings.md)
