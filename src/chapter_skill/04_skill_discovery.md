# 技能发现与注册

前面几节我们学会了定义技能和学习技能。但在多 Agent 系统中，还需要解决一个关键问题：**Agent 如何知道其他 Agent 有什么技能？如何找到并调用合适的技能？**

这就是技能发现与注册机制要解决的问题。

## 为什么需要技能发现？

![企业多 Agent 技能全景图](../svg/chapter_skill_04_agents_skills.svg)

没有技能发现机制，协调 Agent 就不知道该找谁、谁有什么能力。

## 方式一：Agent Card——静态技能声明

### A2A 协议中的技能声明

Google 的 A2A（Agent-to-Agent）协议中，每个 Agent 通过 **Agent Card** 声明自己的技能：

```json
{
  "name": "数据分析 Agent",
  "description": "专业的数据分析与可视化服务",
  "url": "https://data-agent.example.com",
  "version": "2.0",
  "skills": [
    {
      "id": "csv-analysis",
      "name": "CSV 数据分析",
      "description": "对 CSV 文件进行数据清洗、统计分析和趋势识别",
      "input_modes": ["text", "file"],
      "output_modes": ["text", "file", "image"],
      "tags": ["data", "analysis", "csv", "statistics"]
    },
    {
      "id": "data-visualization",
      "name": "数据可视化",
      "description": "根据数据自动选择合适的图表类型并生成可视化",
      "input_modes": ["text", "file"],
      "output_modes": ["image", "file"],
      "tags": ["visualization", "chart", "plot"]
    },
    {
      "id": "report-generation",
      "name": "分析报告生成",
      "description": "基于分析结果生成结构化的 Markdown 或 PDF 报告",
      "input_modes": ["text"],
      "output_modes": ["text", "file"],
      "tags": ["report", "document", "summary"]
    }
  ],
  "authentication": {
    "type": "bearer_token"
  }
}
```

### 技能发现流程

```python
# A2A 技能发现流程
class AgentRegistry:
    """Agent 注册中心"""
    
    def __init__(self):
        self.agents = {}  # agent_url → agent_card
    
    def register(self, agent_url: str):
        """注册 Agent：获取其 Agent Card"""
        card = self._fetch_agent_card(agent_url)
        self.agents[agent_url] = card
        print(f"✅ 注册成功: {card['name']} "
              f"(技能: {[s['name'] for s in card['skills']]})")
    
    def discover_by_skill(self, skill_query: str) -> list:
        """根据技能描述发现合适的 Agent"""
        results = []
        for url, card in self.agents.items():
            for skill in card["skills"]:
                # 匹配技能描述或标签
                if (skill_query.lower() in skill["description"].lower() or
                    any(skill_query.lower() in tag for tag in skill["tags"])):
                    results.append({
                        "agent_name": card["name"],
                        "agent_url": url,
                        "skill": skill
                    })
        return results
    
    def _fetch_agent_card(self, url: str) -> dict:
        """从 Agent 的 well-known 端点获取 Agent Card"""
        import requests
        response = requests.get(f"{url}/.well-known/agent.json")
        return response.json()

# 使用示例
registry = AgentRegistry()
registry.register("https://data-agent.example.com")
registry.register("https://finance-agent.example.com")

# 发现能做"数据分析"的 Agent
agents = registry.discover_by_skill("数据分析")
# → [{"agent_name": "数据分析 Agent", "skill": {...}}]
```

### Agent Card 的最佳实践

![好的与差的技能描述对比](../svg/chapter_skill_04_good_bad_desc.svg)

```
✅ 好的技能描述：
  {
    "name": "CSV 数据分析",
    "description": "对 CSV 格式的表格数据进行数据清洗（缺失值处理、异常值检测）、
                    描述统计（均值、中位数、标准差）、趋势分析和相关性分析",
    "tags": ["csv", "data", "analysis", "statistics", "trend"]
  }

❌ 差的技能描述：
  {
    "name": "分析",
    "description": "分析数据",
    "tags": ["data"]
  }

关键原则：
  1. 描述要具体——说明输入、处理过程、输出
  2. 标签要丰富——覆盖同义词和相关概念
  3. 区分输入/输出模式——文本、文件、图片等
```

---

## 方式二：语义检索——动态技能发现

Agent Card 是静态的，需要预先定义好技能。而**语义检索**可以实现更灵活的动态发现——根据任务描述，在技能库中搜索语义最匹配的技能。

### 基本原理

```
静态发现（关键词匹配）：
  查询："分析一下客户流失率"
  匹配：tags 中包含 "分析" → 找到 3 个 Agent
  问题：不够精确，可能返回无关的"文本分析"Agent

动态发现（语义检索）：
  查询："分析一下客户流失率"
  嵌入查询向量 → 与所有技能描述向量计算相似度
  → 精确匹配到 "客户行为分析" 技能（相似度 0.92）
  → 而不是 "文本情感分析" 技能（相似度 0.45）
```

### 实现

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSkillDiscovery:
    """基于语义检索的技能发现"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.skills = []
        self.embeddings = []
    
    def register_skill(self, skill: dict):
        """注册技能并计算嵌入"""
        # 拼接技能的名称、描述和标签
        text = f"{skill['name']}: {skill['description']}. " \
               f"标签: {', '.join(skill.get('tags', []))}"
        embedding = self.model.encode(text)
        
        self.skills.append(skill)
        self.embeddings.append(embedding)
    
    def discover(self, task: str, top_k: int = 3) -> list:
        """根据任务描述发现最匹配的技能"""
        task_embedding = self.model.encode(task)
        
        # 计算余弦相似度
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = np.dot(task_embedding, emb) / (
                np.linalg.norm(task_embedding) * np.linalg.norm(emb)
            )
            similarities.append((i, sim))
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, sim in similarities[:top_k]:
            results.append({
                "skill": self.skills[idx],
                "similarity": float(sim)
            })
        
        return results

# 使用示例
discovery = SemanticSkillDiscovery()

# 注册技能
discovery.register_skill({
    "name": "客户流失分析",
    "description": "分析客户行为数据，预测流失风险，给出留存策略",
    "agent_url": "https://crm-agent.example.com",
    "tags": ["客户", "流失", "预测", "留存"]
})

discovery.register_skill({
    "name": "文本情感分析",
    "description": "分析文本的情感倾向：正面、负面、中性",
    "agent_url": "https://nlp-agent.example.com",
    "tags": ["NLP", "情感", "文本"]
})

# 发现技能
results = discovery.discover("我想看看哪些客户可能要流失了")
# → [{"skill": {"name": "客户流失分析", ...}, "similarity": 0.92}]
```

---

## 方式三：MCP 与技能生态

在第 12 章中我们将详细介绍 MCP 协议。这里先了解 MCP 如何支撑技能生态：

![MCP 与技能的关系](../svg/chapter_skill_04_mcp_skill.svg)

### MCP 与 A2A 的协作

MCP 与 A2A 的协作关系已在上图中展示：MCP 负责 Agent 与工具/数据之间的垂直扩展，A2A 负责 Agent 之间的水平协作，两者结合实现完整的技能生态。

---

## 技能版本管理

在生产环境中，技能需要版本管理——新版本的技能可能改变了行为，需要兼容性处理：

```python
class VersionedSkillRegistry:
    """支持版本管理的技能注册中心"""
    
    def __init__(self):
        self.skills = {}  # skill_name → {version → skill_definition}
    
    def register(self, name: str, version: str, definition: dict):
        """注册特定版本的技能"""
        if name not in self.skills:
            self.skills[name] = {}
        self.skills[name][version] = definition
    
    def get_skill(self, name: str, version: str = "latest") -> dict:
        """获取技能（支持版本指定）"""
        if name not in self.skills:
            raise ValueError(f"未找到技能: {name}")
        
        if version == "latest":
            versions = sorted(self.skills[name].keys())
            version = versions[-1]
        
        return self.skills[name][version]
    
    def list_versions(self, name: str) -> list:
        """列出技能的所有版本"""
        return sorted(self.skills[name].keys())

# 使用
registry = VersionedSkillRegistry()
registry.register("data_analysis", "1.0", {
    "description": "基础数据分析",
    "capabilities": ["descriptive_stats"]
})
registry.register("data_analysis", "2.0", {
    "description": "高级数据分析",
    "capabilities": ["descriptive_stats", "predictive_modeling", "anomaly_detection"]
})
```

---

## 技能编排：从发现到调用

把所有部分串起来——一个完整的技能编排流程：

```python
class SkillOrchestrator:
    """技能编排器：发现 → 选择 → 调用 → 组合"""
    
    def __init__(self, discovery: SemanticSkillDiscovery, llm):
        self.discovery = discovery
        self.llm = llm
    
    def execute(self, task: str) -> dict:
        """执行任务：自动发现和编排技能"""
        
        # 1. 任务分解
        subtasks = self._decompose_task(task)
        
        # 2. 为每个子任务发现技能
        skill_plan = []
        for subtask in subtasks:
            candidates = self.discovery.discover(subtask, top_k=3)
            best_skill = self._select_best(subtask, candidates)
            skill_plan.append({
                "subtask": subtask,
                "skill": best_skill
            })
        
        # 3. 按顺序执行
        results = []
        for step in skill_plan:
            result = self._invoke_skill(
                step["skill"], 
                step["subtask"],
                context=results  # 传递之前步骤的结果
            )
            results.append(result)
        
        # 4. 整合结果
        return self._combine_results(task, results)
    
    def _decompose_task(self, task: str) -> list:
        """用 LLM 分解任务"""
        prompt = f"将以下任务分解为 2-5 个子任务：\n{task}"
        return self.llm.generate(prompt)
    
    def _select_best(self, subtask: str, candidates: list) -> dict:
        """选择最合适的技能"""
        if not candidates:
            return None
        # 选择相似度最高的
        return candidates[0]["skill"]
```

## 本节小结

| 发现方式 | 机制 | 优点 | 缺点 |
|---------|------|------|------|
| **Agent Card** | 静态声明 | 标准化、可靠 | 需要预先定义 |
| **语义检索** | 向量相似度 | 灵活、智能 | 依赖嵌入质量 |
| **MCP 生态** | 协议发现 | 工具即技能 | 需要 MCP Server |
| **混合方式** | 以上结合 | 最全面 | 复杂度高 |

> 💡 **行业趋势**：2025 年，技能发现正在标准化。Google 的 A2A 协议定义了 Agent Card 中的技能声明格式，Anthropic 的 SKILL.md 定义了技能的描述格式，社区项目 [add-skill](https://add-skill.org/) 提供了跨平台的技能安装工具。未来，Agent 的技能生态可能像今天的 npm/pip 包管理一样，形成标准化的发现、安装和共享体系。

---

*下一节：[5.5 实战：构建可复用的技能系统](./05_practice_skill_system.md)*
