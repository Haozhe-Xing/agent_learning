# 实战：多 Agent 软件开发团队

综合本章所有知识，构建一个完整的多 Agent 软件开发系统，模拟真实的开发团队协作。

## 系统设计

这个项目模拟了一个完整的软件开发团队，包含 6 个角色：产品经理、架构师、开发工程师、测试工程师、DevOps 工程师和文档工程师。每个角色由一个独立的 Agent 节点承担，它们通过**共享状态**（`DevState`）传递工作成果。

### 设计理念

这个系统的设计遵循两个核心原则：

1. **专业分工**：每个 Agent 只负责自己擅长的领域。产品经理不写代码，开发工程师不写测试用例。这种分工让每个 Agent 的 Prompt 可以更专注，输出质量更高。

2. **流水线 + 并行**：工作流并非完全串行——开发完成后，测试、运维和文档三个任务可以并行执行（它们之间没有依赖关系）。LangGraph 天然支持这种并行执行模式。

### 工厂函数模式

代码中使用了 `create_agent_node` 工厂函数来创建 Agent 节点，而不是为每个角色写一个独立的函数。这是因为所有角色的行为模式是相同的（接收上游输出 → 调用 LLM → 返回结果），只有角色名称、任务描述和输入/输出字段不同。工厂函数将这些差异参数化，大幅减少了重复代码。

![多 Agent 软件开发团队流程](../svg/chapter_multi_agent_05_dev_team_flow.svg)

## 完整实现

```python
# dev_team_agent.py
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import json

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
console = Console()

class DevState(TypedDict):
    """开发团队共享状态"""
    original_requirement: str
    product_spec: Optional[str]
    technical_design: Optional[str]
    implementation: Optional[str]
    test_cases: Optional[str]
    deployment_config: Optional[str]
    documentation: Optional[str]
    current_phase: str
    issues: list

def create_agent_node(role: str, task_template: str, input_key: str, output_key: str):
    """工厂函数：创建标准 Agent 节点"""
    
    def agent_node(state: DevState) -> dict:
        console.print(f"\n[bold cyan]👤 {role}[/bold cyan] 开始工作...")
        
        # 获取上下文
        context = state.get(input_key, state.get("original_requirement", ""))
        
        response = llm.invoke([
            HumanMessage(content=f"""你是{role}。

任务：{task_template}

输入：
{context}

请给出专业、详细的输出：""")
        ])
        
        result = response.content
        console.print(f"[green]✅ {role} 完成[/green]")
        console.print(Panel(Markdown(result[:300] + "..." if len(result) > 300 else result), 
                           title=role, border_style="blue", expand=False))
        
        return {
            output_key: result,
            "current_phase": role
        }
    
    return agent_node

# 创建各角色节点
product_manager_node = create_agent_node(
    role="产品经理",
    task_template="""分析需求，输出包含以下内容的产品规格文档：
1. 功能描述（用户故事格式：作为...我想要...以便...）
2. 功能模块列表
3. 非功能性需求（性能、安全等）
4. 验收标准""",
    input_key="original_requirement",
    output_key="product_spec"
)

architect_node = create_agent_node(
    role="系统架构师",
    task_template="""根据产品规格，设计技术方案：
1. 技术栈选择（语言/框架/数据库），并说明理由
2. 系统架构图（文字描述）
3. 数据库表设计（主要表和字段）
4. API 接口设计（端点、方法、参数、响应）
5. 关键实现考虑点""",
    input_key="product_spec",
    output_key="technical_design"
)

developer_node = create_agent_node(
    role="Python开发工程师",
    task_template="""根据技术方案，使用 Python/FastAPI 实现代码：
1. 主要的数据模型（Pydantic models）
2. 核心业务逻辑函数
3. API 路由实现
4. 工具函数

要求：
- 代码完整可运行
- 包含类型注解
- 添加必要注释
- 使用最佳实践""",
    input_key="technical_design",
    output_key="implementation"
)

tester_node = create_agent_node(
    role="QA测试工程师",
    task_template="""为实现代码编写测试用例：
1. pytest 单元测试（覆盖主要功能）
2. 边界条件测试
3. 异常处理测试
4. 集成测试用例描述（不需要实现）

要求代码可以直接运行（pip install pytest httpx fastapi）""",
    input_key="implementation",
    output_key="test_cases"
)

devops_node = create_agent_node(
    role="DevOps工程师",
    task_template="""准备部署配置：
1. Dockerfile（多阶段构建）
2. docker-compose.yml（含数据库、应用服务）
3. 环境变量示例文件(.env.example)
4. 基础的 GitHub Actions CI/CD 配置""",
    input_key="implementation",
    output_key="deployment_config"
)

docs_node = create_agent_node(
    role="技术文档工程师",
    task_template="""撰写 API 文档（Markdown 格式）：
1. 项目概述
2. 快速开始（安装、配置、运行）
3. API 参考（每个接口的详细说明）
4. 代码示例（Python requests 调用示例）
5. 常见问题""",
    input_key="technical_design",
    output_key="documentation"
)

# ============================
# 构建开发工作流图
# ============================

# 工作流图的边定义了角色之间的依赖关系：
# - 产品经理 → 架构师：架构设计需要先有产品规格
# - 架构师 → 开发者：开发需要先有技术方案
# - 开发者 → 测试/运维/文档（并行）：三者都依赖代码实现，但彼此独立
#
# 注意：LangGraph 在遇到一个节点有多条出边时，会并行执行所有目标节点。
# 这意味着 tester、devops、docs 三个节点会同时启动，大幅缩短总执行时间。

workflow = StateGraph(DevState)

# 添加节点
workflow.add_node("product_manager", product_manager_node)
workflow.add_node("architect", architect_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)
workflow.add_node("devops", devops_node)
workflow.add_node("docs", docs_node)

# 串联工作流
workflow.add_edge(START, "product_manager")
workflow.add_edge("product_manager", "architect")
workflow.add_edge("architect", "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("developer", "devops")  # 并行：开发完成后，测试和运维同时开始
workflow.add_edge("developer", "docs")   # 并行：文档也同时编写
workflow.add_edge("tester", END)
workflow.add_edge("devops", END)
workflow.add_edge("docs", END)

dev_team = workflow.compile()

# ============================
# 运行开发团队
# ============================

def develop(requirement: str) -> dict:
    """启动开发流程"""
    console.print(Panel(
        f"[bold]🚀 启动多 Agent 开发团队[/bold]\n"
        f"需求：{requirement}",
        border_style="green"
    ))
    
    initial_state = {
        "original_requirement": requirement,
        "product_spec": None,
        "technical_design": None,
        "implementation": None,
        "test_cases": None,
        "deployment_config": None,
        "documentation": None,
        "current_phase": "初始化",
        "issues": []
    }
    
    result = dev_team.invoke(initial_state)
    
    console.print("\n" + Panel(
        "[bold green]🎉 开发完成！[/bold green]\n\n"
        f"✅ 产品规格：{'完成' if result['product_spec'] else '未完成'}\n"
        f"✅ 技术方案：{'完成' if result['technical_design'] else '未完成'}\n"
        f"✅ 代码实现：{'完成' if result['implementation'] else '未完成'}\n"
        f"✅ 测试用例：{'完成' if result['test_cases'] else '未完成'}\n"
        f"✅ 部署配置：{'完成' if result['deployment_config'] else '未完成'}\n"
        f"✅ API文档：{'完成' if result['documentation'] else '未完成'}",
        border_style="green"
    ))
    
    return result


if __name__ == "__main__":
    result = develop("用户管理系统：包括注册、登录、个人信息修改、密码重置功能")
    
    # 保存到文件
    import os
    os.makedirs("output", exist_ok=True)
    
    for key, value in result.items():
        if value and isinstance(value, str) and key not in ["current_phase"]:
            with open(f"output/{key}.md", "w", encoding="utf-8") as f:
                f.write(f"# {key}\n\n{value}")
    
    print("\n📁 所有文件已保存到 output/ 目录")
```

## 本章小结

多 Agent 协作的核心要点：

| 要素 | 关键实践 |
|------|---------|
| 角色设计 | 专业化、边界清晰 |
| 通信机制 | 共享状态（LangGraph）或消息队列 |
| 架构选择 | Supervisor（推荐）或去中心化 |
| 并行执行 | 无依赖的任务同时运行 |
| 错误处理 | 每个 Agent 独立处理异常 |

---

*下一章：[第12章 Agent 通信协议](../chapter_protocol/README.md)*
