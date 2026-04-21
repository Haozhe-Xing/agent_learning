# 10.2 技能的定义与封装

上一节我们了解了技能的概念。本节深入探讨三种主流的技能封装方式，从最简单到最复杂，每种方式都有其适用场景。

## 方式一：Prompt-based Skill（提示型技能）

这是最简单也是最流行的技能封装方式——**用结构化的 Prompt 将领域知识和行为指南注入 Agent**。

### 基本原理

![普通 Agent vs 有技能的 Agent](../svg/chapter_skill_02_prompt_vs_skill.svg)

### Anthropic Skills 框架

2025 年，Anthropic 开源了 [Agent Skills](https://github.com/anthropics/skills) 框架，用 `SKILL.md` 文件定义技能。这是目前最简洁优雅的提示型技能方案。

一个 Skill 的目录结构：

```
my-skill/
├── SKILL.md          # 技能定义文件（核心）
├── examples/         # 示例文件（可选）
│   ├── example1.md
│   └── example2.md
└── templates/        # 模板文件（可选）
    └── report.md
```

`SKILL.md` 的结构：

```markdown
---
name: data-analyst
description: 专业的数据分析技能，能够自动完成数据清洗、分析和可视化
---

# 数据分析师技能

## 你的角色
你是一名专业的数据分析师。当用户提供数据或提出分析需求时，
你会自动执行完整的分析流程。

## 工作流程
1. **数据理解**：检查数据结构、类型、缺失值
2. **数据清洗**：处理异常值和缺失数据
3. **探索性分析**：计算描述统计、发现数据模式
4. **可视化呈现**：选择合适的图表类型
5. **洞察总结**：提供可操作的业务建议

## 关键规则
- 缺失值超过 30% 的列，优先考虑删除而非填充
- 数值异常值使用 IQR 方法（1.5倍四分位距）检测
- 始终在分析开头提供数据质量报告
- 每个可视化都必须有清晰的标题和说明

## 可视化选择指南
| 数据类型 | 分析目标 | 推荐图表 |
|---------|---------|---------|
| 时间序列 | 趋势 | 折线图 |
| 分类 | 比较 | 柱状图 |
| 数值 | 分布 | 直方图/箱线图 |
| 两个数值 | 关系 | 散点图 |
| 占比 | 构成 | 饼图/环形图 |
```

### 提示型技能的层级机制

技能可以嵌套和组合，形成层级结构：

```
项目级技能/
├── SKILL.md                    # 项目总技能
├── code-review/
│   └── SKILL.md                # 代码审查子技能
├── data-analysis/
│   ├── SKILL.md                # 数据分析子技能
│   └── visualization/
│       └── SKILL.md            # 可视化子技能（更细粒度）
└── report-writing/
    └── SKILL.md                # 报告撰写子技能
```

### 渐进式信息披露

好的技能设计应该遵循**渐进式信息披露**原则——不是一次性把所有知识都塞进上下文，而是按需加载：

```python
class SkillManager:
    """渐进式技能加载管理器"""
    
    def __init__(self):
        self.skills = {}
        self.loaded_skills = set()
    
    def register_skill(self, name: str, skill_path: str):
        """注册技能（不立即加载内容）"""
        self.skills[name] = {
            "path": skill_path,
            "summary": self._extract_summary(skill_path),  # 只加载摘要
        }
    
    def get_skill_summaries(self) -> str:
        """返回所有技能的简短摘要（用于 LLM 决策）"""
        summaries = []
        for name, skill in self.skills.items():
            summaries.append(f"- {name}: {skill['summary']}")
        return "\n".join(summaries)
    
    def load_skill(self, name: str) -> str:
        """按需加载完整技能内容"""
        if name not in self.skills:
            raise ValueError(f"未知技能: {name}")
        
        skill_path = self.skills[name]["path"]
        with open(skill_path, "r") as f:
            content = f.read()
        
        self.loaded_skills.add(name)
        return content
    
    def build_system_prompt(self, base_prompt: str, task: str) -> str:
        """根据任务动态构建系统提示"""
        # 第一步：让 LLM 看到所有技能的摘要
        prompt = f"""{base_prompt}

你具备以下技能：
{self.get_skill_summaries()}

当前任务：{task}

请先判断需要使用哪些技能，然后我会加载详细的技能指南。
"""
        return prompt
```

### 提示型技能的优缺点

| 优点 | 缺点 |
|------|------|
| 零代码，纯文本定义 | 受限于上下文窗口长度 |
| 跨模型通用（GPT、Claude、开源模型） | 复杂逻辑难以精确控制 |
| 易于版本管理（Git） | 模型可能不完全遵循指南 |
| 快速迭代 | 大量技能加载时 Token 成本高 |

---

## 方式二：Code-based Skill（代码型技能）

代码型技能将技能实现为**可执行的代码模块**——不是通过 Prompt 告诉 Agent 怎么做，而是直接提供可运行的代码。

### 基本原理

```python
# 代码型技能：将分析流程封装为可执行代码
class DataAnalysisSkill:
    """数据分析技能"""
    
    def __init__(self):
        self.name = "data_analysis"
        self.description = "自动化数据分析：清洗、统计、可视化、报告生成"
    
    def analyze(self, file_path: str, analysis_type: str = "auto") -> dict:
        """执行完整的数据分析流程"""
        # 1. 加载数据
        df = self._load_data(file_path)
        
        # 2. 数据质量检查
        quality_report = self._check_quality(df)
        
        # 3. 自动分析
        if analysis_type == "auto":
            analysis_type = self._detect_analysis_type(df)
        
        results = self._run_analysis(df, analysis_type)
        
        # 4. 生成可视化
        charts = self._create_visualizations(df, results)
        
        # 5. 生成报告
        report = self._generate_report(quality_report, results, charts)
        
        return {
            "quality_report": quality_report,
            "analysis_results": results,
            "charts": charts,
            "report": report
        }
    
    def _check_quality(self, df) -> dict:
        """数据质量检查"""
        return {
            "total_rows": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "dtypes": df.dtypes.to_dict()
        }
    
    def _detect_analysis_type(self, df) -> str:
        """自动检测分析类型"""
        has_date = any(df[col].dtype == 'datetime64[ns]' for col in df.columns)
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        if has_date and len(numeric_cols) > 0:
            return "time_series"
        elif len(numeric_cols) >= 2:
            return "correlation"
        else:
            return "descriptive"
    
    # ... 更多内部方法
```

### Voyager 的代码技能库

Voyager（2023，NVIDIA）是代码型技能的经典案例。它在 Minecraft 游戏中，让 Agent 将学会的行为保存为 JavaScript 代码技能：

```javascript
// Voyager 自动生成的技能示例
// 技能名：mineWoodLog
// 描述：挖掘木头并收集木材
async function mineWoodLog(bot) {
  // 找到最近的树木
  const woodBlock = bot.findBlock({
    matching: block => block.name.includes('log'),
    maxDistance: 32
  });
  
  if (!woodBlock) {
    bot.chat("附近没有找到树木");
    return false;
  }
  
  // 走到树木旁边
  await bot.pathfinder.goto(woodBlock.position);
  
  // 挖掘木头
  await bot.dig(woodBlock);
  
  bot.chat("成功获取木材！");
  return true;
}
```

技能库的管理：

```python
# Voyager 技能库的核心逻辑（简化版）
class SkillLibrary:
    """Voyager 风格的代码技能库"""
    
    def __init__(self, embedding_model):
        self.skills = {}           # 技能名 → 技能代码
        self.descriptions = {}     # 技能名 → 技能描述
        self.embeddings = {}       # 技能名 → 描述的向量嵌入
        self.embedding_model = embedding_model
    
    def add_skill(self, name: str, code: str, description: str):
        """添加新技能到库中"""
        self.skills[name] = code
        self.descriptions[name] = description
        self.embeddings[name] = self.embedding_model.embed(description)
        print(f"✅ 新技能已添加: {name}")
    
    def retrieve_skills(self, task: str, top_k: int = 5) -> list:
        """根据任务描述检索最相关的技能"""
        task_embedding = self.embedding_model.embed(task)
        
        # 计算与所有技能的相似度
        similarities = {}
        for name, emb in self.embeddings.items():
            similarities[name] = cosine_similarity(task_embedding, emb)
        
        # 返回最相关的 top_k 个技能
        sorted_skills = sorted(similarities.items(), 
                              key=lambda x: x[1], reverse=True)
        
        return [
            {"name": name, "code": self.skills[name], 
             "description": self.descriptions[name]}
            for name, _ in sorted_skills[:top_k]
        ]
```

### Semantic Kernel 的 Plugin 模式

微软的 Semantic Kernel 从一开始就将"技能"（后更名为 Plugin）作为核心概念：

```python
# Semantic Kernel Plugin 示例
from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    """天气查询技能"""
    
    @kernel_function(
        name="get_weather",
        description="获取指定城市的当前天气信息"
    )
    def get_weather(self, city: str) -> str:
        # 调用天气 API
        weather = call_weather_api(city)
        return f"{city}当前天气：{weather['condition']}，" \
               f"温度 {weather['temp']}°C"
    
    @kernel_function(
        name="get_forecast",
        description="获取指定城市未来几天的天气预报"
    )
    def get_forecast(self, city: str, days: int = 3) -> str:
        forecast = call_forecast_api(city, days)
        return format_forecast(forecast)

# 注册到 Kernel
kernel = Kernel()
kernel.add_plugin(WeatherPlugin(), plugin_name="weather")
```

### 代码型技能的优缺点

| 优点 | 缺点 |
|------|------|
| 执行精确、可靠 | 需要编码能力 |
| 可测试、可调试 | 跨平台适配复杂 |
| 性能好（不消耗 LLM Token） | 灵活性不如 Prompt 型 |
| 可以处理复杂的业务逻辑 | 更新和维护成本高 |

---

## 方式三：Workflow-based Skill（工作流型技能）

工作流型技能将技能编排为**有状态的处理流程**——定义节点（步骤）和边（转换条件），构成一个完整的工作流。

### 基本原理

```python
# 使用 LangGraph 定义工作流型技能
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated

class AnalysisState(TypedDict):
    """分析流程的状态"""
    file_path: str
    raw_data: dict
    clean_data: dict
    analysis_results: dict
    charts: list
    report: str
    quality_score: float

def load_data(state: AnalysisState) -> AnalysisState:
    """步骤1：加载数据"""
    df = pd.read_csv(state["file_path"])
    return {"raw_data": df.to_dict()}

def check_quality(state: AnalysisState) -> AnalysisState:
    """步骤2：检查数据质量"""
    df = pd.DataFrame(state["raw_data"])
    missing_ratio = df.isnull().sum().sum() / df.size
    quality_score = 1 - missing_ratio
    return {"quality_score": quality_score}

def should_clean(state: AnalysisState) -> str:
    """条件路由：是否需要清洗？"""
    if state["quality_score"] < 0.9:
        return "clean"      # 质量不够 → 需要清洗
    else:
        return "analyze"     # 质量足够 → 直接分析

def clean_data(state: AnalysisState) -> AnalysisState:
    """步骤3a：数据清洗"""
    df = pd.DataFrame(state["raw_data"])
    df = df.fillna(df.median(numeric_only=True))
    df = df.drop_duplicates()
    return {"clean_data": df.to_dict()}

def analyze(state: AnalysisState) -> AnalysisState:
    """步骤3b/4：数据分析"""
    data = state.get("clean_data") or state["raw_data"]
    df = pd.DataFrame(data)
    results = {
        "mean": df.mean(numeric_only=True).to_dict(),
        "std": df.std(numeric_only=True).to_dict(),
        "correlation": df.corr(numeric_only=True).to_dict()
    }
    return {"analysis_results": results}

def visualize(state: AnalysisState) -> AnalysisState:
    """步骤5：生成可视化"""
    # 生成图表
    return {"charts": ["trend.png", "distribution.png"]}

def generate_report(state: AnalysisState) -> AnalysisState:
    """步骤6：生成报告"""
    report = format_report(state["analysis_results"], state["charts"])
    return {"report": report}

# 构建工作流
workflow = StateGraph(AnalysisState)

# 添加节点
workflow.add_node("load", load_data)
workflow.add_node("check", check_quality)
workflow.add_node("clean", clean_data)
workflow.add_node("analyze", analyze)
workflow.add_node("visualize", visualize)
workflow.add_node("report", generate_report)

# 添加边
workflow.set_entry_point("load")
workflow.add_edge("load", "check")
workflow.add_conditional_edges("check", should_clean, {
    "clean": "clean",
    "analyze": "analyze"
})
workflow.add_edge("clean", "analyze")
workflow.add_edge("analyze", "visualize")
workflow.add_edge("visualize", "report")
workflow.add_edge("report", END)

# 编译为可执行的技能
data_analysis_skill = workflow.compile()

# 使用技能
result = data_analysis_skill.invoke({
    "file_path": "sales.csv"
})
print(result["report"])
```

### 工作流型技能的可视化

![数据分析工作流型技能](../svg/chapter_skill_02_workflow.svg)

### 工作流型技能的优缺点

| 优点 | 缺点 |
|------|------|
| 流程可视化，逻辑清晰 | 架构复杂度较高 |
| 支持条件分支和循环 | 需要学习工作流框架 |
| 可以包含人机协作节点 | 调试比纯代码更难 |
| 天然支持状态管理和错误恢复 | 简单任务可能过度工程化 |

---

## 三种方式的对比与选择

| 维度 | Prompt-based | Code-based | Workflow-based |
|------|-------------|------------|---------------|
| **定义方式** | Markdown / 文本 | Python / JS 代码 | 状态图 / DAG |
| **复杂度** | 低 | 中 | 高 |
| **精确度** | 中（依赖 LLM 理解） | 高 | 高 |
| **灵活性** | 高（自然语言） | 中 | 中 |
| **可测试性** | 低 | 高 | 高 |
| **适用场景** | 知识密集、创意型 | 精确计算、API 集成 | 多步流程、需要审批 |
| **代表框架** | Anthropic Skills | Semantic Kernel | LangGraph |
| **Token 成本** | 高（占用上下文） | 低 | 低 |

### 选择建议

![技能封装方式选择](../svg/chapter_skill_02_decision_tree.svg)

### 混合使用示例

实际项目中，三种方式经常混合使用：

```python
# 混合技能：Prompt + Code + Workflow
class HybridDataAnalystSkill:
    
    # Prompt-based: 注入分析方法论
    SYSTEM_PROMPT = """
    你是一名资深数据分析师。
    分析原则：先看数据质量，再做探索性分析，最后给出建议。
    """
    
    # Code-based: 精确的数据处理
    def clean_data(self, df):
        """代码保证数据清洗的精确性"""
        df = df.dropna(thresh=len(df) * 0.7, axis=1)
        for col in df.select_dtypes(include=['number']):
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5 * iqr) & 
                     (df[col] <= q3 + 1.5 * iqr)]
        return df
    
    # Workflow-based: 多步骤流程编排
    def build_workflow(self):
        """状态图保证流程的可控性"""
        graph = StateGraph(AnalysisState)
        graph.add_node("load", self.load_data)
        graph.add_node("quality_check", self.check_quality)
        graph.add_node("clean", self.clean_data)
        graph.add_node("llm_analyze", self.llm_analyze)  # LLM + Prompt
        graph.add_node("code_visualize", self.visualize)  # 代码生成图表
        # ... 编排
        return graph.compile()
```

## 本节小结

- **Prompt-based Skill** 最适合知识密集型任务——用 Markdown 文件快速定义领域知识和行为指南
- **Code-based Skill** 最适合需要精确控制的任务——用可执行代码实现可靠的技能逻辑
- **Workflow-based Skill** 最适合复杂多步骤流程——用状态图编排带条件分支的工作流
- 实际项目中**三种方式混合使用**是最佳实践

> 💡 **实践建议**：初学者建议从 Prompt-based Skill 开始——用 `SKILL.md` 文件定义你的第一个技能。当需要更精确的控制时，逐步引入 Code-based 和 Workflow-based 技能。

---

*下一节：[9.3 技能学习与获取](./03_skill_learning.md)*
