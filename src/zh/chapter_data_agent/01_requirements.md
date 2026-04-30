# 21.1 需求分析与架构设计

> **本节目标**：设计一个智能数据分析 Agent 的整体方案。

---

## 项目背景

数据分析是很多企业的刚需，但传统数据分析需要掌握 SQL、Python、统计学等技能。我们要构建一个 Agent，让用户用自然语言就能完成数据分析。

**用户说**："帮我分析上个月的销售数据，按区域对比增长率"

**Agent 做**：连接数据库 → 写 SQL 查询 → 分析数据 → 生成可视化图表 → 输出报告

![数据分析 Agent 处理流程](../svg/chapter_data_agent_01_pipeline.svg)

---

## 核心功能

| 功能 | 描述 |
|------|------|
| 自然语言查询 | 用中文描述分析需求，自动转换为 SQL |
| 数据探索 | 自动了解表结构、数据分布 |
| 统计分析 | 描述性统计、趋势分析、对比分析 |
| 可视化 | 自动生成图表 |
| 报告生成 | 整合分析结果为结构化报告 |

---

## 架构设计

```python
from dataclasses import dataclass, field
from enum import Enum

class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"    # 描述性分析
    COMPARATIVE = "comparative"    # 对比分析
    TREND = "trend"               # 趋势分析
    CORRELATION = "correlation"   # 相关性分析

@dataclass
class AnalysisRequest:
    """分析请求"""
    question: str                  # 用户的自然语言问题
    analysis_type: AnalysisType = None  # 自动识别
    target_tables: list[str] = field(default_factory=list)
    time_range: dict = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """分析结果"""
    summary: str                   # 文字总结
    data: dict = field(default_factory=dict)  # 原始数据
    sql_query: str = ""            # 使用的 SQL
    chart_path: str = ""           # 图表文件路径
    insights: list[str] = field(default_factory=list)  # 洞察
```

---

## 技术选型

构建数据分析 Agent 需要从多个维度选择技术栈：

```python
TECH_STACK = {
    "LLM 层": {
        "推荐": "GPT-4o",
        "原因": "SQL 生成准确率最高，代码理解能力强",
        "备选": "GPT-4o-mini（成本敏感场景）、Claude 4（长上下文分析）",
    },
    "数据库连接层": {
        "推荐": "SQLAlchemy",
        "原因": "支持多种数据库（SQLite、PostgreSQL、MySQL）的统一接口",
        "备选": "直接使用各数据库驱动（sqlite3、psycopg2）",
    },
    "数据处理层": {
        "推荐": "pandas",
        "原因": "数据清洗、转换、统计分析的事实标准",
        "备选": "polars（大数据量场景，性能更好）",
    },
    "可视化层": {
        "推荐": "matplotlib + seaborn",
        "原因": "功能全面、文档丰富、图表质量高",
        "备选": "plotly（交互式图表）、pyecharts（中文友好）",
    },
    "Agent 框架": {
        "推荐": "LangGraph",
        "原因": "有状态的分析流程控制、支持人工审核节点",
        "备选": "原生 OpenAI API（简单场景）",
    },
}
```

### 关键依赖安装

```bash
# 数据分析 Agent 的核心依赖
pip install langchain langchain-openai langgraph \
            sqlalchemy pandas matplotlib seaborn \
            tabulate python-dotenv
```

---

## 安全设计考量

数据分析 Agent 直接与数据库交互，安全性是**第一优先级**：

```python
SECURITY_REQUIREMENTS = {
    "SQL 注入防护": {
        "措施": "只允许 SELECT 语句，禁止 INSERT/UPDATE/DELETE/DROP",
        "实现": "正则白名单 + 参数化查询",
        "优先级": "P0（必须）",
    },
    "数据库权限最小化": {
        "措施": "使用只读账号连接数据库",
        "实现": "创建专用 readonly 用户，仅授予 SELECT 权限",
        "优先级": "P0（必须）",
    },
    "查询资源限制": {
        "措施": "防止生成消耗大量资源的查询（如全表扫描、笛卡尔积）",
        "实现": "强制添加 LIMIT、设置查询超时时间",
        "优先级": "P1（重要）",
    },
    "敏感数据脱敏": {
        "措施": "避免将敏感数据（手机号、身份证）发送给 LLM",
        "实现": "查询结果中自动检测并遮蔽敏感字段",
        "优先级": "P1（重要）",
    },
}
```

> ⚠️ **重要提醒**：永远不要让 LLM 生成的 SQL 直接执行。必须经过安全检查层——这是数据分析 Agent 与聊天机器人最大的安全差异。

---

## 完整 Pipeline 设计

![数据分析 Agent 六步 Pipeline](../svg/chapter_data_agent_02_analysis_flow.svg)

将分析流程分为六个阶段，每个阶段有明确的输入和输出（详见上图）。

---

## Agent 核心结构

下面是 `DataAnalysisAgent` 的骨架设计。每个方法代表分析流程中的一个独立阶段，我们将在后续章节中逐一实现。这里先说明每个方法的**设计思路**：

```python
class DataAnalysisAgent:
    """智能数据分析 Agent"""
    
    def __init__(self, llm, db_connection):
        self.llm = llm
        self.db = db_connection
        self.table_schemas = self._load_schemas()
    
    def _load_schemas(self) -> dict:
        """加载数据库表结构
        
        设计思路：
        - 查询 INFORMATION_SCHEMA 获取所有表名、列名、数据类型
        - 将表结构格式化为 LLM 可理解的文本（如 "users(id INT, name TEXT, ..."）
        - 结果缓存到 self.table_schemas，避免每次分析重复查询
        - 这些结构信息将作为 LLM 生成 SQL 时的上下文
        """
        schemas = {}
        # 实际实现见 20.2 节 SafeDatabaseConnector
        return schemas
    
    async def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """执行分析流程——六步 Pipeline"""
        
        # 1. 理解用户意图
        #    将自然语言映射到 AnalysisType（描述、对比、趋势、相关性）
        #    提取时间范围、目标表等关键信息
        intent = await self._understand_intent(request.question)
        
        # 2. 生成 SQL 查询
        #    结合表结构 + 用户意图，让 LLM 生成安全的 SELECT 查询
        #    包含 SQL 注入防护（只允许 SELECT，禁止 DDL/DML）
        sql = await self._generate_sql(request.question, intent)
        
        # 3. 执行查询
        #    通过参数化查询执行，返回结构化数据（dict/DataFrame）
        data = self._execute_query(sql)
        
        # 4. 分析数据
        #    根据分析类型选择不同的分析策略（统计指标、增长率计算等）
        #    让 LLM 从数据中提取关键洞察
        insights = await self._analyze_data(data, request.question)
        
        # 5. 生成可视化
        #    根据分析类型自动选择图表类型（柱状图、折线图、散点图等）
        #    使用 matplotlib 生成图表并保存
        chart_path = self._create_chart(data, intent)
        
        # 6. 生成总结
        #    将数据、洞察、图表整合为自然语言的分析报告
        summary = await self._generate_summary(
            data, insights, request.question
        )
        
        return AnalysisResult(
            summary=summary,
            data=data,
            sql_query=sql,
            chart_path=chart_path,
            insights=insights
        )
```

> **架构要点**：这个六步 Pipeline 的关键设计决策是将"理解"（步骤1-2）和"执行"（步骤3-6）分离。理解阶段由 LLM 驱动，执行阶段由确定性代码完成。这种混合架构既利用了 LLM 的语义理解能力，又保证了数据处理的可靠性和可控性。各个方法的完整实现将在 20.2-20.4 节中逐步展开。

---

## 小结

本节确定了数据分析 Agent 的核心功能和架构。接下来我们将逐步实现各个模块。

---

[21.2 数据连接与查询](./02_data_connection.md)
