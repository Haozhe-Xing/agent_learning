# AutoGPT 与 BabyAGI 的启示

AutoGPT（2023年3月）和 BabyAGI（2023年4月）是最早引起广泛关注的自主 Agent 项目。虽然它们在生产环境中的实用性有限，但其设计理念对整个 Agent 领域产生了深远影响。

![AutoGPT vs BabyAGI 核心循环对比](../svg/chapter_frameworks_01_autogpt_babyagi.svg)

## AutoGPT：自主执行任务的先驱

AutoGPT 的核心理念是：给定一个目标，Agent 自主规划并执行所有步骤。

```python
# AutoGPT 的核心循环（简化版，理解其设计理念）

class AutoGPTLoop:
    """
    模拟 AutoGPT 的核心循环
    展示其设计理念，非真实实现
    """
    
    def __init__(self, goal: str):
        self.goal = goal
        self.memory = []
        self.tools = ["search", "write_file", "read_file", "execute_code"]
    
    def think(self, context: str) -> dict:
        """推理：分析当前状态，决定下一步"""
        from openai import OpenAI
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": f"""你是一个自主 AI Agent，目标是：{self.goal}
                    
你需要：
1. 思考（Thought）：分析当前进展
2. 推理（Reasoning）：为什么这样做
3. 计划（Plan）：接下来的3步
4. 批评（Criticism）：当前方案的风险
5. 行动（Action）：从工具列表选一个立即执行的动作

可用工具：{self.tools}

返回 JSON 格式。"""
                },
                {
                    "role": "user",
                    "content": f"当前上下文：\n{context}\n\n请决定下一步行动："
                }
            ],
            response_format={"type": "json_object"}
        )
        import json
        return json.loads(response.choices[0].message.content)
    
    def execute(self, action: dict) -> str:
        """执行工具动作"""
        tool = action.get("tool", "unknown")
        args = action.get("args", {})
        
        # 模拟执行
        print(f"  执行工具：{tool}({args})")
        return f"{tool} 执行完成，结果：..."
    
    def run(self, max_steps: int = 10):
        """运行主循环"""
        context = f"目标：{self.goal}\n已执行步骤：无"
        
        for step in range(max_steps):
            print(f"\n=== 步骤 {step + 1} ===")
            
            # 思考
            thought = self.think(context)
            print(f"思考：{thought.get('thought', '')}")
            print(f"计划：{thought.get('plan', [])[:1]}")
            
            # 执行
            action = thought.get("action", {})
            if action.get("tool") == "task_complete":
                print("✅ 任务完成！")
                break
            
            result = self.execute(action)
            
            # 更新记忆和上下文
            self.memory.append({"step": step+1, "action": action, "result": result})
            context = f"目标：{self.goal}\n" + "\n".join([
                f"步骤{m['step']}：{m['action'].get('tool')} → {m['result'][:50]}"
                for m in self.memory[-5:]
            ])

# 演示（不实际运行，避免无限循环）
# agent = AutoGPTLoop("写一篇关于 Python 的博客文章并保存")
# agent.run()
```

## AutoGPT 的局限性与教训

AutoGPT 在实践中暴露了几个重要问题：

```python
# 问题1：目标漂移（Goal Drift）
# Agent 在执行过程中可能偏离原始目标
original_goal = "写一篇博客文章"
# 实际执行路径可能是：
# → 搜索大量信息 → 搜索相关工具 → 研究写作技巧 → ... → 忘记写文章

# 问题2：无限循环
# 没有有效的终止条件，Agent 可能永远执行下去
# 解决方案：严格的 max_steps 和 budget 限制

# 问题3：任务分解能力有限
# 自动规划的质量远不如精心设计的 Prompt
# 解决方案：人工辅助规划 + Agent 执行

# 问题4：错误传播
# 早期步骤的小错误可能在后续步骤中放大
# 解决方案：每步验证 + 回滚机制
```

## BabyAGI：任务管理的核心思想

BabyAGI 引入了**任务队列**的概念：

```python
from collections import deque
from openai import OpenAI

client = OpenAI()

class BabyAGI:
    """
    BabyAGI 核心机制：任务队列 + 自动创建子任务
    """
    
    def __init__(self, objective: str):
        self.objective = objective
        self.task_queue: deque = deque()
        self.completed_tasks = []
        self.task_id_counter = 0
    
    def add_task(self, task: str):
        self.task_id_counter += 1
        self.task_queue.append({
            "id": self.task_id_counter,
            "task": task
        })
    
    def execute_task(self, task: str) -> str:
        """执行单个任务"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"你是执行者，目标：{self.objective}"
                },
                {
                    "role": "user",
                    "content": f"完成任务：{task}\n已完成：{self.completed_tasks[-3:]}"
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content
    
    def create_new_tasks(self, task: str, result: str) -> list:
        """基于任务结果创建新的子任务"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""基于以下结果，列出1-2个需要继续执行的子任务（如已完成目标则返回空列表）：
目标：{self.objective}
刚完成任务：{task}
任务结果：{result}
待执行队列：{list(self.task_queue)[:3]}

返回JSON：{{"new_tasks": ["子任务1", "子任务2"]}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        import json
        result_data = json.loads(response.choices[0].message.content)
        return result_data.get("new_tasks", [])
    
    def run(self, initial_task: str, max_tasks: int = 10):
        """运行 BabyAGI 循环"""
        self.add_task(initial_task)
        
        while self.task_queue and len(self.completed_tasks) < max_tasks:
            task = self.task_queue.popleft()
            
            print(f"\n[任务 {task['id']}] {task['task']}")
            
            # 执行
            result = self.execute_task(task["task"])
            self.completed_tasks.append({"task": task["task"], "result": result[:100]})
            print(f"  结果：{result[:100]}")
            
            # 创建新任务
            new_tasks = self.create_new_tasks(task["task"], result)
            for new_task in new_tasks:
                self.add_task(new_task)
                print(f"  → 新增任务：{new_task}")
        
        print(f"\n完成！共执行 {len(self.completed_tasks)} 个任务")

# 测试（控制在小范围内）
agent = BabyAGI("研究 Python 装饰器的主要用法")
agent.run("列举 Python 装饰器的3种常见应用场景", max_tasks=5)
```

## 对现代 Agent 开发的启示

AutoGPT 和 BabyAGI 给我们留下的最重要教训：

```python
# 1. 目标需要清晰和有界
# ❌ "让我们的产品变得更好"
# ✅ "分析用户反馈并列出前5个最常见的投诉"

# 2. 工具必须受限
# ❌ 给 Agent 完全的文件系统访问权限
# ✅ 只给完成任务必要的最小权限

# 3. 终止条件必须明确
# ❌ "一直运行直到完成"
# ✅ "最多执行10步，每步都需要验证进展"

# 4. 人工监督很重要
# 全自动化 Agent 在生产环境风险高
# Human-in-the-Loop 是实际可行的方案
```

---

## 小结

AutoGPT/BabyAGI 的历史价值：
- 证明了 LLM 可以自主执行复杂任务（概念验证）
- 揭示了全自动化 Agent 的局限性（无限循环、错误传播）
- 奠定了任务队列、记忆管理等核心设计模式
- 激发了后续更成熟框架（LangChain、LangGraph）的发展

---

*下一节：[13.2 CrewAI：角色扮演型多 Agent 框架](./02_crewai.md)*
