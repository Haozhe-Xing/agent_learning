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

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
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
    test_timeout_seconds: float
    status: str                  # running / delivered / failed / timed_out / budget_exhausted
    test_status: str             # pending / passed / failed / timed_out / error
    attempts: list[dict]
    failure_report: Optional[dict]


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
        "- 用户故事：作为用户，我希望进行加、乘运算\n"
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
    try:
        design = state["technical_design"]
        if state["test_output"]:
            design += f"\n\n上一轮测试反馈：\n{state['test_output']}"
        code, test_code = dev.develop(design, state["original_requirement"])
        ws = state["workspace_dir"]
        Path(ws, "app.py").write_text(code, encoding="utf-8")
        Path(ws, "app_test.py").write_text(test_code, encoding="utf-8")
    except Exception as exc:
        return {
            "status": "failed",
            "failure_report": {"reason": "generation_error", "message": str(exc)},
            "current_phase": "developer",
        }
    return {
        "implementation": code,
        "current_phase": "developer",
        "iteration": state.get("iteration", 0) + 1,
    }


def _stop_test_processes(proc: subprocess.Popen) -> None:
    """清理独立进程组，包括 pytest 正常退出后留下的子进程。

    POSIX 会话/进程组不是安全沙箱：主动 setsid 的不可信程序可以逃逸。
    其他平台只终止直接子进程；生产环境应使用外部隔离执行器。
    """
    if os.name == "posix":
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                break
            if sig == signal.SIGTERM:
                try:
                    proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
    elif proc.poll() is None:
        proc.kill()
    proc.wait()  # 无论成功或超时，都回收直接子进程。


def tester(state: DevState) -> dict:
    ws = state["workspace_dir"]
    timeout = state["test_timeout_seconds"]
    # 用文件接收输出，避免后台子进程继承 PIPE 后使 communicate 一直等待 EOF。
    with tempfile.TemporaryFile(mode="w+b") as output:
        proc = None
        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "pytest", "app_test.py", "-q"],
                cwd=ws,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
            )
            try:
                returncode = proc.wait(timeout=timeout)
                # pytest: 1 为断言失败，可重试；2—5 为执行/收集/配置错误。
                test_status = "passed" if returncode == 0 else "failed" if returncode == 1 else "error"
                message = "" if returncode == 0 else f"pytest exited with code {returncode}"
            except subprocess.TimeoutExpired:
                returncode, test_status = None, "timed_out"
                message = f"pytest exceeded {timeout:g} seconds"
        except OSError as exc:
            returncode, test_status, message = None, "error", str(exc)
        finally:
            if proc is not None:
                _stop_test_processes(proc)
        output.seek(0)
        test_output = output.read().decode("utf-8", errors="replace")
    attempt = {
        "iteration": state["iteration"], "test_status": test_status,
        "returncode": returncode, "timeout_seconds": timeout,
        "message": message, "output": test_output,
    }
    status = {"timed_out": "timed_out", "error": "failed"}.get(test_status, "running")
    return {
        "test_output": test_output,
        "test_passed": test_status == "passed",
        "test_status": test_status,
        "status": status,
        "attempts": state["attempts"] + [attempt],
        "failure_report": {"reason": test_status, "message": message} if status != "running" else None,
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
    return {"documentation": readme, "current_phase": "docs", "status": "delivered"}


def _deliver(state: DevState, node) -> dict:
    try:
        return node(state)
    except Exception as exc:
        return {
            "status": "failed", "current_phase": node.__name__,
            "failure_report": {"reason": "delivery_error", "message": str(exc)},
        }


def fail(state: DevState) -> dict:
    status = state["status"]
    if status == "running":
        status = "budget_exhausted"
    reason = state.get("failure_report") or {
        "reason": "max_iterations", "message": "Tests did not pass within the iteration budget",
    }
    report = {
        **reason, "status": status, "delivered": False,
        "iteration": state["iteration"], "max_iterations": state["max_iterations"],
        "attempts": state["attempts"],
    }
    Path(state["workspace_dir"], "failure_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return {"status": status, "failure_report": report, "current_phase": "failure"}


def route_after_test(state: DevState) -> str:
    """只有通过才交付；超时/执行错误直接失败，断言失败在预算内重试。"""
    if state.get("status") in {"failed", "timed_out"}:
        return "fail"
    passed = state.get("test_passed")
    hit_cap = state.get("iteration", 0) >= state.get("max_iterations", 3)
    if passed:
        return "deliver"
    if hit_cap:
        return "fail"
    return "fix"


def build_graph(dev: Developer):
    g = StateGraph(DevState)
    g.add_node("product_manager", product_manager)
    g.add_node("architect", architect)
    g.add_node("developer", partial(developer, dev=dev))
    g.add_node("tester", tester)
    g.add_node("devops", partial(_deliver, node=devops))
    g.add_node("docs", partial(_deliver, node=docs))
    g.add_node("failure", fail)

    g.add_edge(START, "product_manager")
    g.add_edge("product_manager", "architect")
    g.add_edge("architect", "developer")
    g.add_conditional_edges("developer", lambda state: "fail" if state["status"] == "failed" else "test", {
        "test": "tester", "fail": "failure",
    })
    g.add_conditional_edges("tester", route_after_test, {
        "deliver": "devops",
        "fix": "developer",
        "fail": "failure",
    })
    g.add_conditional_edges("devops", lambda state: "fail" if state["status"] == "failed" else "docs", {
        "docs": "docs", "fail": "failure",
    })
    g.add_conditional_edges("docs", lambda state: "fail" if state["status"] == "failed" else "end", {
        "end": END, "fail": "failure",
    })
    g.add_edge("failure", END)
    return g.compile()


def develop(requirement: str, dev: Optional[Developer] = None,
            workspace_dir: Optional[str] = None, *, max_iterations: int = 3,
            test_timeout_seconds: float = 30.0) -> dict:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    if (isinstance(test_timeout_seconds, bool) or not isinstance(test_timeout_seconds, (int, float))
            or test_timeout_seconds <= 0):
        raise ValueError("test_timeout_seconds must be a finite positive number")
    try:
        timeout = float(test_timeout_seconds)
    except OverflowError as exc:
        raise ValueError("test_timeout_seconds must be a finite positive number") from exc
    if not math.isfinite(timeout):
        raise ValueError("test_timeout_seconds must be a finite positive number")
    dev = DeterministicDeveloper() if dev is None else dev
    ws = workspace_dir or tempfile.mkdtemp(prefix="devteam_")
    if os.path.lexists(ws) and (not Path(ws).is_dir() or any(Path(ws).iterdir())):
        raise ValueError("workspace_dir must not exist or be an empty directory")
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
        "max_iterations": max_iterations,
        "test_timeout_seconds": timeout,
        "status": "running",
        "test_status": "pending",
        "attempts": [],
        "failure_report": None,
    }, config={"recursion_limit": 2 * max_iterations + 10})
    return dict(result)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", default="算术模块：加法和乘法")
    parser.add_argument("--workspace-dir")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--test-timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    out = develop(args.requirement, workspace_dir=args.workspace_dir,
                  max_iterations=args.max_iterations, test_timeout_seconds=args.test_timeout_seconds)
    print("交付状态：", out["status"], "（已交付）" if out["status"] == "delivered" else "（未交付）")
    print("测试通过：", out["test_passed"])
    print("产物目录：", out["workspace_dir"])
    print(out["test_output"])
    if out["status"] != "delivered":
        print(json.dumps(out["failure_report"], ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
