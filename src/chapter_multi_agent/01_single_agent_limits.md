# 单 Agent 的局限性

理解单 Agent 的瓶颈，才能知道何时需要引入多 Agent 架构。

![单Agent三大局限vs多Agent解决方案](../svg/chapter_multi_agent_01_limits.svg)

## 三个核心限制

```python
# 限制1：Context Window 限制
# 单 Agent 的上下文窗口有限（即使 128K Token 也会在复杂任务中耗尽）

# 示例：分析整个代码库
problem = """
任务：分析 50,000 行代码，找出所有安全漏洞

单 Agent 的困境：
- 无法在单次调用中处理全部代码
- 必须分批处理，但如何保持上下文连贯性？
- 不同批次的分析结果如何整合？
"""

# 限制2：专业知识边界
# 一个 Agent 很难同时成为多个领域的专家

# 示例：全栈项目开发
fullstack_task = """
任务：构建一个完整的 Web 应用

需要的专业知识：
- 前端 React/Vue 开发
- 后端 Python/Node.js 开发
- 数据库设计（SQL/NoSQL）
- DevOps/CI-CD 配置
- 安全审计

单 Agent 的问题：一个 Agent 在所有领域都只有"平均"水平
"""

# 限制3：并行能力
# 单 Agent 本质上是串行的，无法真正并行执行

sequential_time = sum([10, 8, 12, 9, 7])  # 单 Agent：46秒
parallel_time = max([10, 8, 12, 9, 7])    # 多 Agent 并行：12秒
print(f"时间节省：{sequential_time - parallel_time} 秒（{(sequential_time-parallel_time)/sequential_time*100:.0f}%）")
```

## 多 Agent 的优势

```python
# 优势展示：并行处理不同模块

import concurrent.futures
import time
from openai import OpenAI

client = OpenAI()

def single_agent_approach(tasks: list[str]) -> list[str]:
    """单 Agent：串行处理"""
    results = []
    for task in tasks:
        # 每次调用需要等待
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}],
            max_tokens=100
        )
        results.append(response.choices[0].message.content)
    return results

def multi_agent_approach(tasks: list[str]) -> list[str]:
    """多 Agent：并行处理（每个任务一个独立 Agent）"""
    def process_task(task: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": task}],
            max_tokens=100
        )
        return response.choices[0].message.content
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(process_task, tasks))
    
    return results

# 对比测试
tasks = [
    "用一句话描述Python的特点",
    "用一句话描述JavaScript的特点", 
    "用一句话描述Go语言的特点",
    "用一句话描述Rust语言的特点",
    "用一句话描述Java语言的特点",
]

start = time.time()
single_results = single_agent_approach(tasks)
single_time = time.time() - start

start = time.time()
multi_results = multi_agent_approach(tasks)
multi_time = time.time() - start

print(f"单 Agent 耗时：{single_time:.2f}s")
print(f"多 Agent 耗时：{multi_time:.2f}s")
print(f"加速比：{single_time/multi_time:.1f}x")
```

## 什么时候使用多 Agent？

```python
# 决策函数
def should_use_multi_agent(task: dict) -> bool:
    """判断是否需要多 Agent"""
    
    criteria = {
        "需要并行处理": task.get("parallelizable", False),
        "需要多专业领域": len(task.get("domains", [])) > 2,
        "任务复杂度高": task.get("complexity", 0) > 7,
        "时间敏感": task.get("time_sensitive", False),
        "需要互相验证": task.get("requires_verification", False),
    }
    
    # 满足2个以上条件就考虑多 Agent
    met_criteria = sum(criteria.values())
    
    print("评估结果：")
    for criterion, met in criteria.items():
        print(f"  {'✅' if met else '❌'} {criterion}")
    print(f"满足 {met_criteria} 个条件")
    
    return met_criteria >= 2

# 测试
print(should_use_multi_agent({
    "name": "全栈应用开发",
    "parallelizable": True,
    "domains": ["前端", "后端", "数据库", "安全"],
    "complexity": 9,
    "time_sensitive": True,
    "requires_verification": True
}))
```

---

## 小结

使用多 Agent 的场景：
- 任务可以并行化（大幅节省时间）
- 需要多个专业领域（角色专业化）
- 任务超出单个 Context Window
- 需要相互验证（提升准确性）

> 📖 **想深入了解多 Agent 系统的学术前沿？** 请阅读 [14.6 论文解读：多 Agent 系统前沿研究](./06_paper_readings.md)，涵盖 MetaGPT、ChatDev、AutoGen、AgentVerse 等核心论文的深度解读。

---

*下一节：[14.2 多 Agent 通信模式](./02_communication_patterns.md)*
