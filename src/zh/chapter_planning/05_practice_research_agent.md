# 5.5 实战：自动化研究助手 Agent

综合本章所学的规划、推理和反思能力，构建一个能够自主进行研究的 Agent。

> **设计说明**：本项目采用"Plan-then-Execute"的多阶段 Pipeline 架构，而非纯 ReAct 循环。这是因为在研究任务中，各阶段（规划→搜索→分析→质量检查）有明确的先后顺序，Pipeline 模式更易于控制流程和调试。在 Pipeline 内部，每个阶段仍然运用了 ReAct 思想——Agent 根据当前阶段的输出"思考"下一步行动，并在质量检查阶段进行"反思"。这体现了第 5.1 节讨论的"将合适的推理框架应用到合适的场景"原则。

> **前沿定位**：本节的研究助手是 **Deep Research Agent** 的入门形态。真正的 Deep Research Agent 不只是“搜索几次然后总结”，而是能围绕开放问题持续提出子问题、跨来源验证证据、管理引用、识别矛盾，并在多轮研究中逐步收敛结论。

## 从 Search Agent 到 Deep Research Agent

传统搜索 Agent 的目标是“找到答案”；Deep Research Agent 的目标是“形成可信结论”。二者的差异不在于是否联网，而在于是否具备**长程研究流程**。

| 能力维度 | 搜索 Agent | Deep Research Agent |
|---------|------------|---------------------|
| **任务目标** | 回答一个具体问题 | 研究一个开放主题并形成报告 |
| **规划方式** | 一次性生成搜索词 | 动态拆解研究问题，持续补充子问题 |
| **信息处理** | 摘要前几条结果 | 多来源交叉验证、去重、冲突检测 |
| **上下文管理** | 保存搜索结果 | 管理研究笔记、证据卡片、引用链 |
| **质量控制** | 简单检查完整性 | 检查覆盖度、可信度、时效性、反方观点 |
| **输出形式** | 简短回答 | 带引用、结构化论证和不确定性说明的报告 |

可以把 Deep Research Agent 理解为由多个子能力组成的研究流水线：

![Deep Research Agent 研究流水线](../svg/chapter_planning_05_research_pipeline.svg)

本节代码为了教学简洁，只实现其中的核心骨架：规划、搜索、综合、质量检查。你可以在此基础上逐步扩展成完整的 Deep Research Agent。

## 研究助手功能设计

![研究助手 Agent 功能设计](../svg/chapter_planning_05_research_arch.svg)

## 完整实现

```python
import json
import datetime
from openai import OpenAI
import requests

client = OpenAI()

class ResearchAssistant:
    """自动化研究助手"""
    
    def __init__(self):
        self.research_notes = []
        self.sources = []
    
    def _search(self, query: str) -> str:
        """搜索工具（使用 DuckDuckGo）"""
        try:
            url = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": 1}
            response = requests.get(url, params=params, timeout=8)
            data = response.json()
            
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
                if data.get("AbstractURL"):
                    self.sources.append(data["AbstractURL"])
            
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"][:300])
            
            return "\n".join(results) if results else "未找到相关结果"
        except Exception as e:
            return f"搜索失败：{e}"
    
    def _take_notes(self, content: str, source: str = ""):
        """记录研究笔记"""
        self.research_notes.append({
            "content": content,
            "source": source,
            "time": datetime.datetime.now().isoformat()
        })
    
    def research(self, topic: str, depth: str = "standard") -> str:
        """
        执行研究
        
        Args:
            topic: 研究主题
            depth: "quick"=快速概览, "standard"=标准研究, "deep"=深度研究
        """
        
        depth_config = {
            "quick": {"max_searches": 2, "sections": 3},
            "standard": {"max_searches": 4, "sections": 5},
            "deep": {"max_searches": 8, "sections": 7}
        }
        config = depth_config.get(depth, depth_config["standard"])
        
        print(f"\n🔬 开始研究：{topic}")
        print(f"研究深度：{depth}\n")
        
        # ===== 阶段1：规划研究 =====
        print("📋 阶段1：制定研究计划...")
        plan_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""你是一位研究分析师。为以下主题制定研究计划：

主题：{topic}
研究目标：全面理解该主题，生成{config['sections']}个核心章节的报告

请生成JSON格式的研究计划：
{{
  "research_questions": ["核心问题1", "核心问题2", ...],
  "search_queries": ["搜索词1", "搜索词2", ...（最多{config['max_searches']}个）],
  "report_outline": ["章节1标题", "章节2标题", ...]
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        plan = json.loads(plan_response.choices[0].message.content)
        search_queries = plan.get("search_queries", [topic])[:config["max_searches"]]
        report_outline = plan.get("report_outline", [f"{topic}概述"])
        
        print(f"  搜索计划：{len(search_queries)} 个查询")
        print(f"  报告结构：{len(report_outline)} 个章节")
        
        # ===== 阶段2：搜索信息 =====
        print("\n🔍 阶段2：搜索信息...")
        all_findings = []
        
        for i, query in enumerate(search_queries, 1):
            print(f"  搜索 [{i}/{len(search_queries)}]：{query}")
            result = self._search(query)
            
            self._take_notes(result, source=f"搜索：{query}")
            all_findings.append(f"【查询：{query}】\n{result}")
        
        findings_text = "\n\n".join(all_findings)
        
        # ===== 阶段3：分析和综合 =====
        print("\n🧠 阶段3：分析综合...")
        
        analysis_response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": f"""基于以下研究资料，对主题"{topic}"进行深度分析。

研究资料：
{findings_text[:4000]}

报告大纲：{report_outline}

请按大纲生成完整的研究报告，要求：
1. 每个章节有实质性内容（200-400字）
2. 包含具体的数据、案例或观点
3. 在报告末尾给出结论和建议
4. 使用Markdown格式"""
                }
            ]
        )
        
        report = analysis_response.choices[0].message.content
        
        # ===== 阶段4：质量检查 =====
        print("\n✅ 阶段4：质量检查...")
        
        review_response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"""简要评估以下研究报告的质量（JSON格式）：

主题：{topic}
报告（前1000字）：{report[:1000]}

评估：
{{
  "completeness_score": 1-10,
  "accuracy_indicators": "高/中/低",
  "missing_aspects": ["遗漏点1"],
  "overall_quality": "优秀/良好/一般"
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        
        review = json.loads(review_response.choices[0].message.content)
        
        # 生成最终报告
        final_report = f"""# 研究报告：{topic}

> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
> 研究深度：{depth}
> 质量评分：{review.get('completeness_score', 'N/A')}/10
> 信息来源：{len(self.research_notes)} 条

---

{report}

---

## 研究说明

- 本报告基于 {len(search_queries)} 次网络搜索
- 信息截止日期：{datetime.datetime.now().strftime('%Y-%m-%d')}
- 建议结合最新资料进行验证
"""
        
        print(f"\n📄 报告生成完成！")
        print(f"质量：{review.get('overall_quality', 'N/A')} | "
              f"完整性：{review.get('completeness_score', 'N/A')}/10")
        
        return final_report


# 使用示例
assistant = ResearchAssistant()

report = assistant.research(
    topic="大语言模型在软件开发中的应用",
    depth="standard"
)

# 保存报告
filename = f"research_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.md"
with open(filename, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n📁 报告已保存到：{filename}")
```

## 关键代码解读

这个研究助手虽然代码不长，但已经具备 Deep Research Agent 的雏形：

- **规划阶段**：先生成 `research_questions` 和 `search_queries`，避免边搜边迷路。
- **信息收集阶段**：每次搜索都写入 `research_notes`，形成可追溯的研究轨迹。
- **综合阶段**：不是简单拼接搜索结果，而是按 `report_outline` 重组信息。
- **质量检查阶段**：引入第二次模型调用评估覆盖度，模拟研究员的自审流程。

如果要把它升级为生产级 Deep Research Agent，建议补上四个模块：

| 模块 | 作用 | 实现要点 |
|------|------|----------|
| **证据卡片** | 保存事实、来源 URL、发布时间、可信度 | 每条结论都能追溯到来源 |
| **反向搜索** | 主动寻找反例和不同观点 | 避免只采纳支持性证据 |
| **引用检查** | 验证报告中的引用是否真实支持结论 | 防止“引用幻觉” |
| **研究状态机** | 控制研究阶段切换 | 防止无限搜索或过早总结 |

Deep Research Agent 的关键不是“多搜”，而是**让每一次搜索都服务于一个明确的研究缺口**。这也是长程规划能力在真实 Agent 应用中的典型落点。

## 运行研究助手

```bash
pip install openai python-dotenv requests rich
python research_agent.py
```

示例输出：
```markdown
# 研究报告：大语言模型在软件开发中的应用

> 生成时间：2024-03-15 14:30
> 研究深度：standard
> 质量评分：8/10

## 1. 概述
...

## 2. 代码生成与补全
...

## 3. 代码审查与 Bug 检测
...
```

## 小结

本节完成了一个自动化研究助手，并展示了 Deep Research Agent 的基础架构：

- ✅ 研究计划生成：把开放主题拆成问题和搜索词
- ✅ 多轮信息收集：记录搜索结果和来源
- ✅ 分析综合：按大纲生成结构化报告
- ✅ 质量检查：对完整性和遗漏点进行自审
- ✅ 可扩展方向：证据卡片、反向搜索、引用检查、研究状态机

真正的 Deep Research Agent 是“长程规划 + Web/文档工具 + 证据治理 + 质量评估”的组合体。它会在后续的 Web Agent、上下文工程、评估与安全章节中继续展开。

---

*下一节：[5.6 Plan-and-Execute 与 Test-time Compute Scaling](./07_plan_and_execute.md)*