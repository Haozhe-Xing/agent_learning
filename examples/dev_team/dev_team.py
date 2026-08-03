"""多 Agent 软件开发团队（可运行版）。

与"只让每个角色用 LLM 吐一段文字"的演示有本质区别：
- 开发工程师把代码真正写到磁盘（app.py + app_test.py）
- 测试工程师用 pytest 真正跑测试并收集结果
- 测试不通过就回到开发工程师重试（测试驱动的闭环）

为了让图"不依赖 API key 也能跑、也能测"，代码生成器（Developer）是可
注入的：默认用 DeterministicDeveloper 写出一份已知正确的模块；真实场景
再换成调用 LLM 的实现（见参考底座 reference-agent）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from functools import partial
from typing import Optional, Protocol, TypedDict

from langgraph.graph import START, END, StateGraph


class DevState(TypedDict):
    original_requirement: str
    workspace_dir: str            # 各角色把产物写到这个目录
    product_spec: Optional[str]
    technical_design: Optional[str]
    implementation: Optional[str]
    test_output: Optional[str]    # pytest 的真实输出（不是"假装通过"）
    test_passed: Optional[bool]   # 测试是否通过（客观判据）
    deployment_config: Optional[str]
    documentation: Optional[str]
    current_phase: str
    iteration: int
    max_iterations: int


class Developer(Protocol):
    """代码生成器：输入设计，输出 (源码, 测试源码)。"""
    def develop(self, design: str, requirement: str) -> tuple[str, str]: ...


class DeterministicDeveloper:
    """离线开发者：写出一份已知能通过测试的模块，保证图可跑可测。"""

    def develop(self, design: str, requirement: str) -> tuple[str, str]:
        code = (
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n"
            "\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n"
        )
        test_code = (
            "from app import add, multiply\n"
            "\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
            "    assert add(-1, 1) == 0\n"
            "\n"
            "def test_multiply():\n"
            "    assert multiply(2, 3) == 6\n"
        )
        return code, test_code


class LLMDeveloper:
    """真实开发者：调用 LLM 生成代码与测试（需注入 llm 与解析逻辑）。"""

    def __init__(self, llm):
        self.llm = llm

    def develop(self, design: str, requirement: str) -> tuple[str, str]:
        # 把 design + requirement 交给 LLM，从返回中解析出 code / test_code。
        raise NotImplementedError(
            "注入你的代码生成提示与解析；离线演示用 DeterministicDeveloper"
        )


# ---------- 节点 ----------

def product_manager(state: DevState) -> dict:
    spec = (
        f"# 产品规格\n\n需求：{state['original_requirement']}\n\n"
        "- 用户故事：作为用户，我希望进行加、减、乘运算\n"
        "- 验收标准：add / multiply 计算结果正确\n"
    )
    return {"product_spec": spec, "current_phase": "product_manager"}


def architect(state: DevState) -> dict:
    design = (
        "# 技术方案\n\n- 语言：Python\n"
        "- 模块：app.py 暴露 add / multiply\n"
        "- 测试：app_test.py 用 pytest 覆盖\n"
    )
    return {"technical_design": design, "current_phase": "architect"}


def developer(state: DevState, dev: Developer) -> dict:
    code, test_code = dev.develop(state["technical_design"], state["original_requirement"])
    ws = state["workspace_dir"]
    with open(os.path.join(ws, "app.py"), "w", encoding="utf-8") as f:
        f.write(code)
    with open(os.path.join(ws, "app_test.py"), "w", encoding="utf-8") as f:
        f.write(test_code)
    return {
        "implementation": code,
        "current_phase": "developer",
        "iteration": state.get("iteration", 0) + 1,
    }


def tester(state: DevState) -> dict:
    # 注意：这里真正运行 pytest，而不是"假装测试通过"
    ws = state["workspace_dir"]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "app_test.py", "-q"],
        cwd=ws,
        capture_output=True,
        text=True,
    )
    return {
        "test_output": proc.stdout + proc.stderr,
        "test_passed": proc.returncode == 0,
        "current_phase": "tester",
    }


def devops(state: DevState) -> dict:
    dockerfile = (
        "FROM python:3.12-slim\nWORKDIR /app\nCOPY . .\n"
        "RUN pip install pytest\nCMD [\"python\", \"-m\", \"pytest\", \"-q\"]\n"
    )
    with open(os.path.join(state["workspace_dir"], "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile)
    return {"deployment_config": dockerfile, "current_phase": "devops"}


def docs(state: DevState) -> dict:
    readme = (
        "# 自动生成的项目\n\n由多 Agent 开发团队产出。\n\n"
        "## 测试\n\n```bash\npytest -q\n```\n"
    )
    with open(os.path.join(state["workspace_dir"], "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    return {"documentation": readme, "current_phase": "docs"}


def route_after_test(state: DevState) -> str:
    """测试通过后交付；未通过且未达上限则回到开发重试。"""
    passed = state.get("test_passed")
    hit_cap = state.get("iteration", 0) >= state.get("max_iterations", 3)
    if passed or hit_cap:
        return "deliver"
    return "fix"


def build_graph(dev: Developer):
    g = StateGraph(DevState)
    g.add_node("product_manager", product_manager)
    g.add_node("architect", architect)
    g.add_node("developer", partial(developer, dev=dev))
    g.add_node("tester", tester)
    g.add_node("devops", devops)
    g.add_node("docs", docs)

    g.add_edge(START, "product_manager")
    g.add_edge("product_manager", "architect")
    g.add_edge("architect", "developer")
    g.add_edge("developer", "tester")
    g.add_conditional_edges("tester", route_after_test, {
        "deliver": "devops",
        "fix": "developer",
    })
    g.add_edge("devops", "docs")
    g.add_edge("docs", END)
    return g.compile()


def develop(requirement: str, dev: Optional[Developer] = None,
            workspace_dir: Optional[str] = None) -> dict:
    dev = dev or DeterministicDeveloper()
    ws = workspace_dir or tempfile.mkdtemp(prefix="devteam_")
    os.makedirs(ws, exist_ok=True)
    app = build_graph(dev)
    result = app.invoke({
        "original_requirement": requirement,
        "workspace_dir": ws,
        "product_spec": None,
        "technical_design": None,
        "implementation": None,
        "test_output": None,
        "test_passed": None,
        "deployment_config": None,
        "documentation": None,
        "current_phase": "init",
        "iteration": 0,
        "max_iterations": 3,
    })
    return dict(result)


if __name__ == "__main__":
    out = develop("用户管理系统：注册、登录、信息修改")
    print("测试通过：", out["test_passed"])
    print("产物目录：", out["workspace_dir"])
    print(out["test_output"])
