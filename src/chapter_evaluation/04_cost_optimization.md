# 成本控制与性能优化

> **本节目标**：学会在保证 Agent 质量的前提下，有效控制 API 调用成本和提升响应速度。

---

## Agent 的成本构成

![Agent 成本优化策略矩阵](../svg/chapter_evaluation_04_cost_strategies.svg)

运行一个 Agent 的成本主要来自三个方面：

| 成本类型 | 来源 | 占比 |
|---------|------|------|
| LLM API 调用 | 每次推理消耗的 Token | 60-80% |
| 向量数据库 | 嵌入计算 + 检索查询 | 10-20% |
| 基础设施 | 服务器、存储、网络 | 10-20% |

一个简单的计算：如果你的 Agent 每次对话平均消耗 5000 个 Token（含输入和输出），使用 GPT-4o，成本约为 $0.02。每天处理 10000 次对话，月费用约 $6000。

---

## 策略一：智能模型路由

不是所有问题都需要最强大（也最贵）的模型。用便宜模型处理简单问题，复杂问题才用高端模型：

```python
from langchain_openai import ChatOpenAI

class SmartRouter:
    """智能模型路由器 —— 根据问题复杂度选择模型"""
    
    def __init__(self):
        # 快速便宜的模型处理简单任务
        self.fast_model = ChatOpenAI(
            model="gpt-4o-mini", temperature=0
        )
        # 强大模型处理复杂任务
        self.power_model = ChatOpenAI(
            model="gpt-4o", temperature=0
        )
    
    def classify_complexity(self, question: str) -> str:
        """判断问题复杂度"""
        # 简单规则判断（也可以用 LLM 判断，但会增加成本）
        simple_indicators = [
            len(question) < 50,           # 短问题通常简单
            "?" in question and question.count("?") == 1,  # 单个问题
            any(w in question for w in ["你好", "谢谢", "是什么"]),
        ]
        
        complex_indicators = [
            len(question) > 200,          # 长问题通常复杂
            "分析" in question or "比较" in question,
            "代码" in question or "实现" in question,
            question.count("?") > 2,      # 多个问题
        ]
        
        simple_score = sum(simple_indicators)
        complex_score = sum(complex_indicators)
        
        if complex_score >= 2:
            return "complex"
        elif simple_score >= 2:
            return "simple"
        else:
            return "medium"
    
    def route(self, question: str) -> ChatOpenAI:
        """选择合适的模型"""
        complexity = self.classify_complexity(question)
        
        if complexity == "simple":
            return self.fast_model    # ~$0.15/1M tokens
        else:
            return self.power_model   # ~$2.50/1M tokens

# 使用示例
router = SmartRouter()
model = router.route("Python 的列表推导式怎么用？")
response = model.invoke("Python 的列表推导式怎么用？")
```

---

## 策略二：语义缓存

如果用户问了类似的问题，直接返回之前的结果，不用再调 API：

```python
import hashlib
import json
import time
import numpy as np
from pathlib import Path

class SemanticCache:
    """基于语义相似度的缓存"""
    
    def __init__(
        self,
        cache_dir: str = ".agent_cache",
        similarity_threshold: float = 0.92,
        ttl: int = 3600  # 缓存过期时间（秒）
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.threshold = similarity_threshold
        self.ttl = ttl
        self.cache = self._load_cache()
    
    def _load_cache(self) -> list[dict]:
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return []
    
    def _save_cache(self):
        with open(self.cache_dir / "cache.json", "w") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
    
    def get(self, query: str, query_embedding: list[float]) -> str | None:
        """尝试从缓存中获取答案"""
        now = time.time()
        
        for entry in self.cache:
            # 检查是否过期
            if now - entry["timestamp"] > self.ttl:
                continue
            
            # 计算语义相似度
            similarity = self._cosine_similarity(
                query_embedding, entry["embedding"]
            )
            
            if similarity >= self.threshold:
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                self._save_cache()
                return entry["response"]
        
        return None
    
    def put(
        self,
        query: str,
        query_embedding: list[float],
        response: str
    ):
        """将结果存入缓存"""
        self.cache.append({
            "query": query,
            "embedding": query_embedding,
            "response": response,
            "timestamp": time.time(),
            "hit_count": 0
        })
        
        # 限制缓存大小
        if len(self.cache) > 1000:
            # 移除最旧的、命中率最低的条目
            self.cache.sort(
                key=lambda x: (x["hit_count"], x["timestamp"])
            )
            self.cache = self.cache[-500:]  # 保留最有价值的一半
        
        self._save_cache()
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

---

## 策略三：Prompt 压缩

在不损失关键信息的情况下，减少发送给 LLM 的 Token 数：

```python
class PromptCompressor:
    """Prompt 压缩器"""
    
    def compress_history(
        self,
        messages: list[dict],
        max_messages: int = 10,
        max_tokens: int = 2000
    ) -> list[dict]:
        """压缩对话历史"""
        if len(messages) <= max_messages:
            return messages
        
        # 策略：保留首尾，压缩中间
        # 1. 始终保留系统消息
        system_msgs = [m for m in messages if m["role"] == "system"]
        
        # 2. 保留最近的消息
        recent = messages[-max_messages:]
        
        # 3. 将中间的消息压缩为摘要
        middle = messages[len(system_msgs):-max_messages]
        if middle:
            summary = self._summarize_messages(middle)
            summary_msg = {
                "role": "system",
                "content": f"[之前对话的摘要：{summary}]"
            }
            return system_msgs + [summary_msg] + recent
        
        return system_msgs + recent
    
    def _summarize_messages(self, messages: list[dict]) -> str:
        """将多条消息压缩为摘要"""
        topics = set()
        for msg in messages:
            # 提取关键词作为主题
            content = msg.get("content", "")
            if len(content) > 20:
                topics.add(content[:50] + "...")
        
        return f"用户讨论了以下话题：{'、'.join(list(topics)[:5])}"
    
    def remove_redundancy(self, prompt: str) -> str:
        """移除提示中的冗余信息"""
        lines = prompt.split("\n")
        seen = set()
        result = []
        
        for line in lines:
            normalized = line.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(line)
            elif not normalized:
                result.append(line)  # 保留空行
        
        return "\n".join(result)
```

---

## 策略四：成本监控与预警

```python
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class CostTracker:
    """API 调用成本追踪器"""
    
    # 各模型的价格（每百万 Token，2026-03 数据，请以官方最新定价为准）
    PRICING = {
        "gpt-4o":      {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-5":       {"input": 10.00, "output": 30.00},
    }
    
    daily_budget: float = 50.0  # 每日预算（美元）
    monthly_budget: float = 1000.0
    
    # 记录
    daily_cost: float = 0.0
    monthly_cost: float = 0.0
    call_log: list = field(default_factory=list)
    model_usage: dict = field(default_factory=lambda: defaultdict(
        lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
    ))
    
    def record_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> dict:
        """记录一次 API 调用"""
        pricing = self.PRICING.get(model, {"input": 5.0, "output": 15.0})
        
        cost = (
            input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000
        )
        
        self.daily_cost += cost
        self.monthly_cost += cost
        
        # 更新模型级别统计
        usage = self.model_usage[model]
        usage["calls"] += 1
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["cost"] += cost
        
        # 检查预算
        warnings = []
        if self.daily_cost > self.daily_budget * 0.8:
            warnings.append(
                f"⚠️ 日预算已使用 {self.daily_cost/self.daily_budget:.0%}"
            )
        if self.monthly_cost > self.monthly_budget * 0.8:
            warnings.append(
                f"⚠️ 月预算已使用 {self.monthly_cost/self.monthly_budget:.0%}"
            )
        
        return {
            "cost": cost,
            "daily_total": self.daily_cost,
            "warnings": warnings
        }
    
    def get_report(self) -> str:
        """生成成本报告"""
        report = "📊 成本报告\n"
        report += f"{'='*40}\n"
        report += f"日成本: ${self.daily_cost:.4f} / ${self.daily_budget:.2f}\n"
        report += f"月成本: ${self.monthly_cost:.4f} / ${self.monthly_budget:.2f}\n\n"
        
        report += "各模型使用统计：\n"
        for model, usage in self.model_usage.items():
            report += f"  {model}:\n"
            report += f"    调用次数: {usage['calls']}\n"
            report += f"    输入 Token: {usage['input_tokens']:,}\n"
            report += f"    输出 Token: {usage['output_tokens']:,}\n"
            report += f"    费用: ${usage['cost']:.4f}\n"
        
        return report
```

---

## 性能优化技巧清单

| 技巧 | 节省效果 | 实现难度 | 说明 |
|------|---------|---------|------|
| 模型路由 | 40-60% | ⭐⭐ | 简单问题用便宜模型 |
| 语义缓存 | 20-50% | ⭐⭐⭐ | 重复问题直接返回 |
| Prompt 压缩 | 10-30% | ⭐⭐ | 减少输入 Token |
| 批量处理 | 20-40% | ⭐ | 合并多个请求 |
| 流式响应 | 不省钱，但体验好 | ⭐ | 降低用户等待感 |
| 预生成 | 节省实时成本 | ⭐⭐⭐ | 热门问题提前生成 |

---

## 小结

| 概念 | 说明 |
|------|------|
| 模型路由 | 根据问题复杂度选择合适的模型 |
| 语义缓存 | 相似问题复用之前的回答 |
| Prompt 压缩 | 保留关键信息，减少 Token 数 |
| 成本监控 | 实时追踪费用，超预算预警 |

> **下一节预告**：在生产环境中，光有性能还不够——你还需要能"看见" Agent 内部发生了什么。

---

[下一节：13.5 可观测性：日志、追踪与监控 →](./05_observability.md)
