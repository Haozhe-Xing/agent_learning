# 18.7 框架补充：CrewAI（角色扮演型多 Agent 框架）

> 🎭 *CrewAI 用"角色+任务+团队"的简洁抽象，让多 Agent 协作非常直观。*

---

CrewAI 是一个专为多 Agent 协作设计的框架，核心思路是用"角色扮演"建模协作：每个 Agent 有明确的 **role（角色）/ goal（目标）/ backstory（背景故事）**，像一支分工明确的团队那样工作。自 2024 年推出以来，它成为最受欢迎的多 Agent 框架之一，并在 2025 年引入了 **Flows** 这一事件驱动工作流特性。

![CrewAI 核心架构：Agent-Task-Crew](../svg/chapter_frameworks_02_crewai.svg)

## 核心抽象：Agent + Task + Crew

CrewAI 围绕三个概念构建：

| 抽象 | 含义 | 你的工作 |
|------|------|---------|
| **Agent** | 一个角色（如研究员、编辑），由 role/goal/backstory 定义"人设" | 写好角色 Prompt，必要时挂工具 |
| **Task** | 一个具体任务，可声明依赖（基于上一任务的输出） | 写好任务描述与期望产出 |
| **Crew** | 一支团队，按 `Process`（sequential / hierarchical）编排执行 | 把 Agent 和 Task 装进去 `kickoff` |

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="资深研究员", goal="收集最新准确信息",
                  backstory="10 年经验，注重数据准确性", verbose=True)
writer = Agent(role="内容编辑", goal="把研究转化为易读文章",
              backstory="擅长把技术内容写给人看", verbose=True)

research_task = Task(description="研究主题：{topic}，收集定义/趋势/场景/挑战",
                     expected_output="详细研究报告", agent=researcher)
write_task = Task(description="基于研究报告写一篇技术文章",
                  expected_output="Markdown 文章", agent=writer,
                  context=[research_task])  # 依赖研究任务输出

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task],
            process=Process.sequential, verbose=True)
result = crew.kickoff(inputs={"topic": "LangGraph 在生产环境的应用"})
```

> 💡 **直觉理解**：CrewAI 把"多 Agent 协作"翻译成了人类最熟悉的组织语言——**招人（Agent）、派活（Task）、组队（Crew）**。它的强项不是复杂控制流，而是让"角色分工 + 流水线"这件事变得极其省心：`backstory` 写得好，角色之间的协作质量就高。

## 两种执行流程

| 流程 | 含义 | 适用 |
|------|------|------|
| `Process.sequential` | 顺序执行，任务按依赖链流转 | 明确的流水线（研究→写作→审查） |
| `Process.hierarchical` | 引入管理者 Agent 动态分配任务 | 需要灵活调度、任务间依赖不确定 |

## Flows：从"角色协作"到"精确编排"

2025 年引入的 **CrewAI Flows** 补足了 Crew 的短板——当流程里有**条件分支、循环、需要精确控制执行顺序**时，用 Python 代码以事件驱动方式编排（装饰器 `@start` / `@listen` / `@router`）。它甚至可以在 Flow 里嵌入 Crew，实现"确定流程 + 角色协作"的混合。

```python
from crewai.flow.flow import Flow, listen, start, router

class ArticleFlow(Flow):
    @start()
    def choose_topic(self):
        self.state.topic = "Agent 开发最佳实践"
        return self.state.topic
    @listen(choose_topic)
    def research(self, topic): ...          # 调研究 Crew
    @router(research)
    def check_quality(self, research):
        return "rewrite" if too_short else "publish"  # 条件路由
```

**Crew vs Flow 选择**：任务分工明确、Agent 可自主协作 → **Crew**；需要精确控制流程、条件分支、循环 → **Flow**。

## CrewAI vs LangGraph

| 维度 | CrewAI | LangGraph |
|------|--------|-----------|
| 上手难度 | 低，声明式、代码量少 | 较高，需定义 State/Node/Edge |
| 适用场景 | 角色分工明确；Flow 支持事件驱动编排 | 需精确控制流、复杂循环、状态管理 |
| 状态管理 | 较弱 | 强（checkpoint + store） |
| 人在回路 / 调试 | 有限 | 原生支持 interrupt/resume + 可视化 |
| 选型直觉 | 快速原型、内容/业务流水线 | 生产级高可靠、复杂有状态工作流 |

> 💡 **选型建议**：角色分工清晰 → CrewAI（Crew 模式）；复杂控制流/状态 → LangGraph；流程编排+多 Agent → CrewAI（Flow）；快速原型 → CrewAI；生产高可靠 → LangGraph。

---

## 小结

CrewAI 用"角色 + 任务 + 团队"的简洁抽象，让多 Agent 协作非常直观。2025 年引入的 **Flows** 进一步补齐了工作流编排能力——从简单的顺序任务到带条件分支/循环的完整工作流都能覆盖。它和 LangGraph 不是竞争关系：前者胜在"像组队一样简单"，后者胜在"像状态机一样精确"。

---

*上一节：[18.6 实战：多 Agent 软件开发团队](../chapter_18_multi_agent/05_practice_dev_team.md)*  
*下一节：[第18章 多 Agent 协作 章节首页](../chapter_18_multi_agent/README.md)*
