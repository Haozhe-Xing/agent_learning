# 如何选择合适的框架？

框架选择是 Agent 项目成功的关键决策之一。随着 2025 年 OpenAI Agents SDK 的发布和各框架的快速迭代，选择变得更加丰富。本节提供系统的决策框架。

![Agent 框架选择决策指南](../svg/chapter_frameworks_05_choose.svg)

## 框架能力对比矩阵

```python
framework_comparison = {
    "框架": [
        "LangChain", "LangGraph", "CrewAI",
        "AutoGen", "OpenAI Agents SDK", "Dify", "Coze/n8n"
    ],
    
    "学习曲线": ["中等", "较高", "低", "中等", "低", "低", "很低"],
    
    "多Agent支持": [
        "有限", "原生支持", "核心特性",
        "核心特性", "原生支持", "支持", "有限"
    ],
    
    "工作流复杂度": [
        "线性", "复杂循环", "顺序/Flow",
        "事件驱动", "Handoff 交接", "可视化", "可视化"
    ],
    
    "MCP 支持": [
        "社区集成", "社区集成", "社区集成",
        "社区集成", "原生支持", "插件", "不支持"
    ],
    
    "生产就绪": ["高", "高", "中", "中", "高", "中", "低"],
    
    "最适场景": [
        "RAG/通用链",
        "有状态工作流",
        "角色协作任务",
        "代码生成/对话",
        "轻量生产Agent",
        "快速原型",
        "非技术用户"
    ]
}

# 打印对比
for key, values in framework_comparison.items():
    print(f"\n{key}：")
    for framework, value in zip(framework_comparison["框架"], values):
        if key != "框架":
            print(f"  {framework}: {value}")
```

## 决策树

```python
def choose_framework(requirements: dict) -> str:
    """根据需求选择框架（2026 年更新版）"""
    
    # 非技术团队 → 低代码
    if not requirements.get("technical_team"):
        return "Dify 或 Coze（低代码平台）"
    
    # 需要代码自动执行 → AutoGen
    if requirements.get("code_execution"):
        return "AutoGen 0.4"
    
    # 轻量 Agent、快速上线 → OpenAI Agents SDK
    if (requirements.get("lightweight") and 
        not requirements.get("complex_control_flow")):
        return "OpenAI Agents SDK"
    
    # 角色分工明确的多 Agent → CrewAI
    if (requirements.get("multi_agent") and 
        not requirements.get("complex_control_flow")):
        return "CrewAI"
    
    # 复杂状态管理/循环/Human-in-Loop → LangGraph
    if (requirements.get("complex_control_flow") or
        requirements.get("human_in_the_loop") or
        requirements.get("stateful_workflow")):
        return "LangGraph"
    
    # 标准 RAG/单 Agent → LangChain
    return "LangChain"


# 测试决策
scenarios = [
    {
        "name": "企业知识库问答",
        "technical_team": True,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "自动化软件开发助手",
        "technical_team": True,
        "multi_agent": True,
        "code_execution": True,
        "complex_control_flow": True,
        "lightweight": False
    },
    {
        "name": "内容创作团队",
        "technical_team": True,
        "multi_agent": True,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "客服自动化（业务配置）",
        "technical_team": False,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "快速构建工具调用 Agent",
        "technical_team": True,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": True
    }
]

print("框架选择建议：\n")
for scenario in scenarios:
    name = scenario["name"]  # 使用 [] 读取而非 pop()，避免修改原始字典
    # 构造不含 name 的需求字典传给决策函数
    requirements = {k: v for k, v in scenario.items() if k != "name"}
    framework = choose_framework(requirements)
    print(f"场景：{name}")
    print(f"推荐：{framework}\n")
```

## 实际项目的框架策略

```python
# 大多数生产项目会混合使用多个框架

class HybridAgentSystem:
    """
    实际生产系统的典型架构（2025-2026）：
    - LangGraph 管理复杂工作流状态
    - OpenAI Agents SDK 处理轻量 Agent 逻辑
    - LangChain 处理 RAG 和链式调用
    - MCP 统一工具接口
    - 自定义代码处理业务逻辑
    """
    
    def __init__(self):
        # LangGraph 负责工作流
        from langgraph.graph import StateGraph
        self.workflow = None  # 用 LangGraph 构建
        
        # LangChain 负责 RAG
        from langchain_community.vectorstores import Chroma
        self.knowledge_base = None  # 用 LangChain 构建
        
        # OpenAI Agents SDK 负责轻量 Agent
        # from agents import Agent, Runner
        self.agents = {}
        
        # 自定义工具（可通过 MCP 标准化）
        self.tools = {}
    
    def build(self):
        """组合各框架构建完整系统"""
        # 1. 用 LangChain 建立知识库
        # 2. 用 LangGraph 建立工作流
        # 3. 用 OpenAI Agents SDK 创建轻量 Agent
        # 4. 用 MCP 标准化工具接口
        pass

# 建议：不要被任何单一框架绑定
# 理解各框架的优势，按需组合
```

## 最终建议

```
选择框架的核心原则：

1. 从简单开始
   先尝试 OpenAI Agents SDK 或 LangChain + 直接 API 调用
   够用就不要引入复杂框架

2. 根据瓶颈升级
   发现需要状态管理 → 引入 LangGraph
   需要多角色协作 → 考虑 CrewAI
   需要代码执行 → 考虑 AutoGen

3. 拥抱标准协议
   用 MCP 统一工具接口，降低框架切换成本
   关注 A2A 协议发展，为 Agent 间互操作做准备

4. 保持框架无关的代码
   业务逻辑不要与框架强耦合
   工具函数保持通用性

5. 重视调试与可观测性
   生产环境首选有良好日志和观测性的方案
   LangSmith、Dify 等都提供了较好的可观测能力

6. 社区和生态
   选择活跃维护的框架（查看 GitHub 活跃度）
   2025 年最活跃：LangGraph、CrewAI、OpenAI Agents SDK
```

---

## 本章小结

主流框架一览：

| 框架 | 核心优势 | 推荐场景 |
|------|---------|---------|
| LangChain | 生态丰富，RAG 强大 | 通用 Agent，快速开发 |
| LangGraph | 状态管理，复杂工作流 | 生产级有状态 Agent |
| CrewAI | 简单的多 Agent + Flows | 角色分工明确的任务 |
| AutoGen 0.4 | 事件驱动，代码执行 | 编程自动化任务 |
| OpenAI Agents SDK | 轻量、MCP 原生 | 快速构建生产 Agent |
| Dify/Coze | 低代码可视化 | 非技术团队快速验证 |

---

*下一章：[第14章 多 Agent 协作](../chapter_multi_agent/README.md)*
