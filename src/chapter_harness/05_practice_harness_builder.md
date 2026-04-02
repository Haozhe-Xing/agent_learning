# 9.5 实战：构建你的第一个 Harness 系统

> 🛠️ *"学习 Harness Engineering 最好的方式，就是亲手为你现有的 Agent 搭建一套，然后看着它在一周内将失败率降低 50%。"*

---

## 目标

本节将带你从零开始，为一个典型的 **Python 编程 Agent** 搭建完整的 Harness 系统。

完成后，你的 Agent 将具备：

- ✅ 自动监控上下文利用率，防止上下文焦虑
- ✅ 强制执行 Plan-Build-Verify-Fix 工作流
- ✅ 工具权限分层，防止危险操作
- ✅ 死循环检测，自动切换策略
- ✅ 可观测性仪表盘，实时监控 Agent 行为

**完整代码**：约 400 行 Python，可直接运行。

---

## 项目结构

```
harness_demo/
├── harness/
│   ├── __init__.py
│   ├── context_manager.py     # 上下文生命周期管理
│   ├── tool_registry.py       # 工具注册与权限控制
│   ├── validation_gate.py     # 自验证循环
│   ├── loop_detector.py       # 死循环检测
│   └── observability.py       # 可观测性
├── agent.py                   # Agent 主体（使用 Harness）
├── AGENTS.md                  # Agent 宪法
└── demo.py                    # 演示脚本
```

---

## Step 1：编写 AGENTS.md

首先为演示项目编写 `AGENTS.md`：

```markdown
# AGENTS.md — Harness Demo Project

## 项目概述
这是一个用于演示 Harness Engineering 的 Python 项目。

## 技术栈
- Python 3.11
- FastAPI（Web 框架）
- pytest（测试框架）
- ruff（Lint 工具）

## 强制工作流程
完成任何代码修改后，必须按顺序执行：
1. `pytest tests/ -v --tb=short`（运行测试）
2. `ruff check src/ --fix`（Lint 检查并自动修复）
3. 若有测试失败，修复后重新执行步骤 1

## 架构约束
- src/api/ → 只能调用 src/services/
- src/services/ → 只能调用 src/models/ 和 src/repositories/
- 禁止在 api 层直接执行数据库查询

## 禁止操作
- 禁止修改现有测试文件（除非明确修复测试 bug）
- 禁止硬编码任何 API Key 或密码
- 禁止删除 tests/ 目录下的任何文件
```

---

## Step 2：上下文管理器

```python
# harness/context_manager.py

import time
from dataclasses import dataclass, field
from typing import Optional
import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """统计 token 数量"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # 粗略估算：平均每 4 个字符 1 个 token
        return len(text) // 4


@dataclass
class ContextStats:
    """上下文统计信息"""
    total_tokens: int
    max_tokens: int
    utilization: float
    message_count: int
    status: str  # "healthy", "warning", "danger"
    recommendation: str


class HarnessContextManager:
    """
    Harness 上下文管理器
    
    功能：
    1. 实时监控上下文利用率
    2. 自动触发渐进式压缩
    3. 提供健康报告
    """
    
    COMPRESSION_THRESHOLD = 0.40
    DANGER_THRESHOLD = 0.70
    
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self._compression_count = 0
    
    def add_message(self, role: str, content: str) -> None:
        """添加消息，并在必要时触发压缩"""
        self.messages.append({"role": role, "content": content})
        
        stats = self.get_stats()
        if stats.utilization >= self.COMPRESSION_THRESHOLD:
            self._auto_compress(stats)
    
    def get_stats(self) -> ContextStats:
        """获取当前上下文统计"""
        total = sum(
            count_tokens(m.get("content", ""))
            for m in self.messages
        )
        util = total / self.max_tokens
        
        if util < self.COMPRESSION_THRESHOLD:
            status, recommendation = "healthy", "正常，无需操作"
        elif util < self.DANGER_THRESHOLD:
            status, recommendation = "warning", "建议清理旧工具输出"
        else:
            status, recommendation = "danger", "⚠️ 立即触发完整压缩"
        
        return ContextStats(
            total_tokens=total,
            max_tokens=self.max_tokens,
            utilization=util,
            message_count=len(self.messages),
            status=status,
            recommendation=recommendation,
        )
    
    def _auto_compress(self, stats: ContextStats) -> None:
        """自动渐进式压缩"""
        # 第一步：轻量压缩——清除旧工具结果
        self._clear_old_tool_results()
        self._compression_count += 1
        
        # 检查是否仍然超限
        new_stats = self.get_stats()
        if new_stats.utilization >= self.COMPRESSION_THRESHOLD:
            # 第二步：完整压缩（需要外部 LLM 调用，这里用简化实现）
            self._truncate_middle()
            self._compression_count += 1
    
    def _clear_old_tool_results(self) -> None:
        """轻量压缩：清除较旧的工具输出"""
        cutoff = len(self.messages) - 8
        for i in range(cutoff):
            msg = self.messages[i]
            if msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                self.messages[i] = {
                    "role": "tool",
                    "content": "[工具输出已归档以节省上下文空间]",
                    "tool_call_id": msg.get("tool_call_id"),
                }
    
    def _truncate_middle(self) -> None:
        """保留头部（系统消息）和尾部（最近对话），压缩中间部分"""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        recent_msgs = self.messages[-6:]
        
        # 用占位符替代中间历史
        self.messages = system_msgs + [
            {
                "role": "system",
                "content": "[早期对话历史已压缩。关键决策：见上方系统消息。]"
            }
        ] + recent_msgs
    
    def get_messages(self) -> list[dict]:
        """获取当前消息列表（供 LLM API 调用使用）"""
        return self.messages.copy()
```

---

## Step 3：工具注册表

```python
# harness/tool_registry.py

from enum import IntEnum
from typing import Callable, Any
from functools import wraps
import json
import subprocess
import os
from pathlib import Path


class PermissionLevel(IntEnum):
    READ_ONLY = 1
    WRITE_SAFE = 2
    WRITE_DESTRUCTIVE = 3


@dataclass
class Tool:
    name: str
    func: Callable
    description: str
    permission: PermissionLevel
    parameters: dict  # JSON Schema 格式
    idempotent: bool = True


class HarnessToolRegistry:
    """
    工具注册表：实现权限分层和强类型约束
    """
    
    def __init__(self, workspace: str, agent_role: str = "code_writer"):
        self.workspace = Path(workspace).resolve()
        self.agent_role = agent_role
        self.tools: dict[str, Tool] = {}
        self._audit_log: list[dict] = []
        
        # 注册默认工具集
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """注册编程 Agent 所需的标准工具集"""
        
        # === 只读工具 ===
        self.register(
            name="read_file",
            func=self._read_file,
            description="读取文件内容。返回文件的文本内容。",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径（相对于工作目录）"},
                    "offset": {"type": "integer", "description": "起始行号（可选，默认 0）"},
                    "limit": {"type": "integer", "description": "读取行数（可选，默认全部）"},
                },
                "required": ["path"],
            },
        )
        
        self.register(
            name="list_files",
            func=self._list_files,
            description="列出目录中的文件和子目录。",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "目录路径（默认为工作目录）"},
                },
                "required": [],
            },
        )
        
        self.register(
            name="search_content",
            func=self._search_content,
            description="在文件中搜索指定内容（支持正则表达式）。",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "搜索模式（正则表达式）"},
                    "path": {"type": "string", "description": "搜索路径（默认为工作目录）"},
                },
                "required": ["pattern"],
            },
        )
        
        # === 写入工具（安全） ===
        self.register(
            name="write_file",
            func=self._write_file,
            description="写入文件内容（会先备份原文件）。",
            permission=PermissionLevel.WRITE_SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"},
                    "content": {"type": "string", "description": "要写入的内容"},
                },
                "required": ["path", "content"],
            },
            idempotent=False,
        )
        
        # === 执行工具 ===
        self.register(
            name="run_tests",
            func=self._run_tests,
            description="运行 pytest 测试套件，返回测试结果摘要。",
            permission=PermissionLevel.WRITE_SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "测试模式（默认运行全部）"},
                },
                "required": [],
            },
        )
        
        self.register(
            name="run_linter",
            func=self._run_linter,
            description="运行 ruff 代码检查，返回 Lint 问题列表。",
            permission=PermissionLevel.WRITE_SAFE,
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
    
    def register(
        self,
        name: str,
        func: Callable,
        description: str,
        permission: PermissionLevel,
        parameters: dict,
        idempotent: bool = True,
    ) -> None:
        """注册工具"""
        # 包装函数以添加审计日志
        wrapped_func = self._wrap_with_audit(func, name, permission)
        
        self.tools[name] = Tool(
            name=name,
            func=wrapped_func,
            description=description,
            permission=permission,
            parameters=parameters,
            idempotent=idempotent,
        )
    
    def _wrap_with_audit(
        self, func: Callable, tool_name: str, permission: PermissionLevel
    ) -> Callable:
        """包装工具函数，添加审计日志"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            entry = {
                "timestamp": time.time(),
                "tool": tool_name,
                "permission": permission.name,
                "args": str(args)[:100],
                "kwargs": str(kwargs)[:100],
            }
            try:
                result = func(*args, **kwargs)
                entry["status"] = "success"
                return result
            except Exception as e:
                entry["status"] = "error"
                entry["error"] = str(e)
                raise
            finally:
                self._audit_log.append(entry)
        return wrapper
    
    def call(self, tool_name: str, **kwargs) -> Any:
        """调用工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在：{tool_name}。可用工具：{list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        return tool.func(**kwargs)
    
    def get_tool_definitions(self) -> list[dict]:
        """返回 OpenAI API 格式的工具定义列表"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
            }
            for tool in self.tools.values()
        ]
    
    # === 工具实现 ===
    
    def _safe_path(self, path: str) -> Path:
        """确保路径在工作目录内（防止路径遍历）"""
        full_path = (self.workspace / path).resolve()
        if not str(full_path).startswith(str(self.workspace)):
            raise ValueError(f"禁止访问工作目录外的文件：{path}")
        return full_path
    
    def _read_file(self, path: str, offset: int = 0, limit: int = None) -> str:
        full_path = self._safe_path(path)
        if not full_path.exists():
            return f"错误：文件不存在：{path}"
        
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if limit:
            lines = lines[offset:offset + limit]
        else:
            lines = lines[offset:]
        
        return "\n".join(f"{offset + i + 1}:{line}" for i, line in enumerate(lines))
    
    def _list_files(self, path: str = ".") -> str:
        full_path = self._safe_path(path)
        if not full_path.is_dir():
            return f"错误：不是目录：{path}"
        
        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(self.workspace)
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{rel_path}")
        
        return "\n".join(items) if items else "（空目录）"
    
    def _search_content(self, pattern: str, path: str = ".") -> str:
        full_path = self._safe_path(path)
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, str(full_path)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout[:3000] if result.stdout else "未找到匹配内容"
    
    def _write_file(self, path: str, content: str) -> str:
        full_path = self._safe_path(path)
        
        # 备份原文件（如果存在）
        if full_path.exists():
            backup_path = full_path.with_suffix(full_path.suffix + ".bak")
            backup_path.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")
        
        # 确保父目录存在
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"✅ 已写入：{path}（{len(content)} 字节）"
    
    def _run_tests(self, pattern: str = "") -> str:
        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        if pattern:
            cmd.append(pattern)
        
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(self.workspace), timeout=120
        )
        
        # 返回摘要而非完整输出（节省上下文）
        output = result.stdout + result.stderr
        lines = output.split("\n")
        
        # 提取关键行：失败的测试和最终统计
        key_lines = [l for l in lines if 
                     "PASSED" in l or "FAILED" in l or "ERROR" in l or 
                     "passed" in l or "failed" in l or "error" in l]
        
        return "\n".join(key_lines[-20:])  # 最多返回最后 20 行关键信息
    
    def _run_linter(self) -> str:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--output-format=text"],
            capture_output=True, text=True,
            cwd=str(self.workspace), timeout=60
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return "✅ 无 Lint 问题"
        return output[:2000]  # 限制输出长度
    
    def get_audit_summary(self) -> str:
        """返回工具调用审计摘要"""
        if not self._audit_log:
            return "无工具调用记录"
        
        counts = {}
        errors = []
        for entry in self._audit_log:
            counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
            if entry["status"] == "error":
                errors.append(f"  - {entry['tool']}: {entry.get('error', 'unknown')}")
        
        lines = ["工具调用统计："]
        for tool, count in sorted(counts.items()):
            lines.append(f"  {tool}: {count} 次")
        
        if errors:
            lines.append("\n调用错误：")
            lines.extend(errors)
        
        return "\n".join(lines)
```

---

## Step 4：验证门控

```python
# harness/validation_gate.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, bool]  # 每项检查的结果
    failures: list[str]
    
    @property
    def failure_report(self) -> str:
        if not self.failures:
            return "所有验证通过"
        return "\n".join(f"❌ {f}" for f in self.failures)


class HarnessValidationGate:
    """
    验证门控：在 Agent 声称完成任务前，
    强制执行一系列检查
    """
    
    def __init__(self, tool_registry):
        self.tools = tool_registry
    
    def validate(
        self,
        task_description: str,
        modified_files: list[str],
    ) -> ValidationResult:
        """
        运行完整验证流程
        
        检查项：
        1. 测试套件通过
        2. Lint 无错误
        3. 未删除测试文件
        4. 修改了预期的文件
        """
        checks = {}
        failures = []
        
        # 检查 1：运行测试
        print("  🧪 运行测试...")
        test_output = self.tools.call("run_tests")
        test_passed = "failed" not in test_output.lower() and "error" not in test_output.lower()
        checks["tests_passed"] = test_passed
        if not test_passed:
            failures.append(f"测试失败:\n{test_output[:500]}")
        
        # 检查 2：Lint 检查
        print("  📋 运行 Lint 检查...")
        lint_output = self.tools.call("run_linter")
        lint_passed = "无 Lint 问题" in lint_output or lint_output.strip() == ""
        checks["lint_passed"] = lint_passed
        if not lint_passed:
            failures.append(f"Lint 问题:\n{lint_output[:300]}")
        
        # 检查 3：检查测试文件完整性（防作弊）
        print("  🛡️ 检查测试文件完整性...")
        test_integrity = self._check_test_integrity(modified_files)
        checks["test_integrity"] = test_integrity
        if not test_integrity:
            failures.append("⚠️ 检测到测试文件被删除或显著减少！这是不允许的。")
        
        passed = len(failures) == 0
        
        if passed:
            print("  ✅ 所有验证通过！")
        else:
            print(f"  ❌ {len(failures)} 项验证失败")
        
        return ValidationResult(
            passed=passed,
            checks=checks,
            failures=failures,
        )
    
    def _check_test_integrity(self, modified_files: list[str]) -> bool:
        """检查测试文件是否被违规删除或大量减少"""
        for file_path in modified_files:
            if "test" in file_path.lower():
                # 读取文件内容检查是否有大量删除
                try:
                    content = self.tools.call("read_file", path=file_path)
                    line_count = len(content.split("\n"))
                    # 如果测试文件现在少于 5 行，很可能有问题
                    if line_count < 5 and "test" in file_path.lower():
                        return False
                except Exception:
                    pass
        return True
```

---

## Step 5：死循环检测

```python
# harness/loop_detector.py

from collections import defaultdict
from dataclasses import dataclass
import time


@dataclass
class LoopWarning:
    file_path: str
    edit_count: int
    suggestion: str


class HarnessLoopDetector:
    """
    死循环检测器
    
    检测：同一文件被反复修改（可能陷入"尝试-失败-再尝试"循环）
    响应：注入"换个思路"的建议，而不是强制中断
    """
    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.edit_history: dict[str, list[float]] = defaultdict(list)
    
    def record_edit(self, file_path: str) -> Optional[LoopWarning]:
        """
        记录一次文件编辑。
        如果超过阈值，返回警告（否则返回 None）。
        """
        now = time.time()
        self.edit_history[file_path].append(now)
        
        # 只看最近 10 分钟内的编辑
        recent = [t for t in self.edit_history[file_path] if now - t < 600]
        self.edit_history[file_path] = recent
        
        if len(recent) > self.threshold:
            return LoopWarning(
                file_path=file_path,
                edit_count=len(recent),
                suggestion=self._generate_suggestion(file_path, len(recent)),
            )
        return None
    
    def _generate_suggestion(self, file_path: str, count: int) -> str:
        return f"""
⚠️ 循环检测警告：你已经对 `{file_path}` 进行了 {count} 次修改。

反复修改同一文件通常表明：
1. 问题的根本原因不在这个文件（请检查调用它的地方）
2. 你的修改方向可能有误（请重新阅读任务要求）
3. 存在其他文件的依赖问题（请用 search_content 查找相关代码）

建议：
- 暂停修改此文件
- 运行测试，仔细阅读错误信息
- 用 search_content 搜索与错误相关的代码
- 如果仍然无法解决，请描述遇到的具体问题
"""
    
    def get_report(self) -> str:
        """返回编辑历史报告"""
        if not self.edit_history:
            return "无编辑记录"
        
        lines = ["文件编辑历史："]
        for path, timestamps in sorted(self.edit_history.items()):
            recent = [t for t in timestamps if time.time() - t < 600]
            if recent:
                status = "⚠️ 警告" if len(recent) > self.threshold else "✅ 正常"
                lines.append(f"  {status} {path}: 最近10分钟 {len(recent)} 次编辑")
        
        return "\n".join(lines)
```

---

## Step 6：整合 Harness Agent

```python
# agent.py

import json
import time
from openai import OpenAI
from harness.context_manager import HarnessContextManager
from harness.tool_registry import HarnessToolRegistry
from harness.validation_gate import HarnessValidationGate
from harness.loop_detector import HarnessLoopDetector


SYSTEM_PROMPT = """
你是一个专业的 Python 编程助手，遵循 Harness Engineering 最佳实践。

## 工作流程（强制）
完成每项编码任务时，必须按以下步骤执行：

### 步骤 1：规划
- 分析任务要求
- 列出需要修改的文件
- 制定具体的修改方案

### 步骤 2：实现
- 按计划逐项实现
- 每次修改文件后，简要描述做了什么

### 步骤 3：验证（强制，不可跳过！）
- 调用 run_tests 运行测试
- 调用 run_linter 检查代码风格
- 对照任务要求核查每一项

### 步骤 4：修复（如有问题）
- 修复验证发现的所有问题
- 重新执行步骤 3

只有当步骤 3 完全通过，才能说"任务完成"。

## 项目规范
详见 AGENTS.md（使用 read_file 工具读取）。
"""


class HarnessAgent:
    """
    使用 Harness 系统的编程 Agent
    """
    
    def __init__(self, workspace: str, api_key: str = None, model: str = "gpt-4o"):
        self.workspace = workspace
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
        # 初始化 Harness 组件
        self.context_manager = HarnessContextManager(max_tokens=128_000)
        self.tool_registry = HarnessToolRegistry(workspace)
        self.validation_gate = HarnessValidationGate(self.tool_registry)
        self.loop_detector = HarnessLoopDetector(threshold=3)
        
        # 加载系统提示
        self.context_manager.add_message("system", SYSTEM_PROMPT)
        
        # 注入环境上下文
        env_context = self._build_env_context()
        self.context_manager.add_message("system", env_context)
        
        # 跟踪修改的文件（用于验证）
        self._modified_files: list[str] = []
        
        print(f"✅ Harness Agent 初始化完成")
        print(f"   工作目录：{workspace}")
        print(f"   模型：{model}")
        print(f"   可用工具：{len(self.tool_registry.tools)} 个")
    
    def _build_env_context(self) -> str:
        """构建环境上下文（渐进式披露：给目录而非完整内容）"""
        try:
            # 获取顶层目录结构
            import os
            items = os.listdir(self.workspace)
            structure = "\n".join(f"  - {item}" for item in sorted(items[:20]))
        except Exception:
            structure = "  （无法读取）"
        
        return f"""
## 工作环境

**工作目录**：{self.workspace}

**顶层目录结构**：
{structure}

**可用工具**：
{chr(10).join(f"  - {name}: {tool.description[:60]}" for name, tool in self.tool_registry.tools.items())}

**提示**：使用 list_files 和 read_file 工具探索项目结构。
使用 read_file 读取 AGENTS.md 了解项目规范。
"""
    
    def execute(self, task: str, max_iterations: int = 20) -> dict:
        """
        执行任务（带完整 Harness 保护）
        
        Returns:
            {
                "success": bool,
                "result": str,
                "iterations": int,
                "validation": ValidationResult,
                "context_stats": ContextStats,
            }
        """
        print(f"\n{'='*60}")
        print(f"🎯 任务：{task}")
        print(f"{'='*60}")
        
        self.context_manager.add_message("user", task)
        
        iteration = 0
        last_error = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- 迭代 {iteration}/{max_iterations} ---")
            
            # 显示上下文健康状态
            stats = self.context_manager.get_stats()
            status_icon = {"healthy": "🟢", "warning": "🟡", "danger": "🔴"}[stats.status]
            print(f"上下文状态：{status_icon} {stats.utilization:.1%} ({stats.total_tokens} tokens)")
            
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.context_manager.get_messages(),
                tools=self.tool_registry.get_tool_definitions(),
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            
            # 检查是否有工具调用
            if message.tool_calls:
                self.context_manager.add_message("assistant", 
                    json.dumps([tc.model_dump() for tc in message.tool_calls]))
                
                # 执行工具调用
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  🔧 调用工具：{tool_name}({list(tool_args.keys())})")
                    
                    # 死循环检测（针对写文件操作）
                    if tool_name == "write_file":
                        file_path = tool_args.get("path", "")
                        warning = self.loop_detector.record_edit(file_path)
                        
                        if file_path not in self._modified_files:
                            self._modified_files.append(file_path)
                        
                        if warning:
                            # 注入循环警告
                            print(f"  ⚠️ 循环检测：{file_path} 已被编辑 {warning.edit_count} 次")
                            self.context_manager.add_message("system", warning.suggestion)
                    
                    # 执行工具
                    try:
                        result = self.tool_registry.call(tool_name, **tool_args)
                    except Exception as e:
                        result = f"工具执行错误：{str(e)}"
                    
                    # 将结果添加到上下文
                    self.context_manager.add_message("tool", str(result)[:2000])
                
            else:
                # 没有工具调用——Agent 在输出文本响应
                content = message.content or ""
                self.context_manager.add_message("assistant", content)
                print(f"  💬 Agent：{content[:200]}...")
                
                # 检查是否声称任务完成
                completion_signals = [
                    "任务完成", "task complete", "已完成", "完成了", 
                    "successfully completed", "done"
                ]
                if any(signal in content.lower() for signal in completion_signals):
                    print("\n🔍 检测到任务完成信号，执行强制验证...")
                    
                    validation = self.validation_gate.validate(
                        task_description=task,
                        modified_files=self._modified_files,
                    )
                    
                    if validation.passed:
                        print("✅ 验证通过！任务成功完成。")
                        return {
                            "success": True,
                            "result": content,
                            "iterations": iteration,
                            "validation": validation,
                            "context_stats": self.context_manager.get_stats(),
                            "tool_audit": self.tool_registry.get_audit_summary(),
                        }
                    else:
                        # 验证失败：注入失败信息，要求修复
                        print(f"❌ 验证失败，要求修复：")
                        print(validation.failure_report)
                        
                        fix_prompt = f"""
验证失败，任务尚未真正完成！请修复以下问题：

{validation.failure_report}

请：
1. 仔细阅读上面的错误信息
2. 修复所有问题
3. 重新运行测试和 Lint 确认通过
4. 通过所有验证后再声明任务完成
"""
                        self.context_manager.add_message("user", fix_prompt)
        
        # 超出最大迭代次数
        print(f"\n⚠️ 超出最大迭代次数 ({max_iterations})")
        return {
            "success": False,
            "result": f"任务在 {max_iterations} 次迭代后未完成",
            "iterations": iteration,
            "validation": None,
            "context_stats": self.context_manager.get_stats(),
            "tool_audit": self.tool_registry.get_audit_summary(),
        }
```

---

## Step 7：运行演示

```python
# demo.py

import os
from agent import HarnessAgent

def main():
    # 配置
    workspace = os.path.join(os.path.dirname(__file__), "sample_project")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # 创建 Harness Agent
    agent = HarnessAgent(
        workspace=workspace,
        api_key=api_key,
        model="gpt-4o",
    )
    
    # 执行任务
    task = """
    请在 src/utils/math_helpers.py 中实现以下函数：
    
    1. `factorial(n: int) -> int`：计算阶乘（n >= 0），n < 0 时抛出 ValueError
    2. `fibonacci(n: int) -> int`：返回第 n 个斐波那契数（0-indexed），n < 0 时抛出 ValueError
    
    要求：
    - 包含完整的类型注解
    - 包含 Google 风格的文档字符串
    - 在 tests/test_math_helpers.py 中编写单元测试，覆盖正常情况和边界情况
    """
    
    result = agent.execute(task)
    
    # 显示结果
    print("\n" + "="*60)
    print("📊 执行报告")
    print("="*60)
    print(f"状态：{'✅ 成功' if result['success'] else '❌ 失败'}")
    print(f"迭代次数：{result['iterations']}")
    
    if result.get("context_stats"):
        stats = result["context_stats"]
        print(f"最终上下文利用率：{stats.utilization:.1%}")
    
    if result.get("tool_audit"):
        print(f"\n{result['tool_audit']}")
    
    if result.get("validation"):
        v = result["validation"]
        print(f"\n验证结果：{'✅ 通过' if v.passed else '❌ 失败'}")
        if v.failures:
            print(f"失败原因：{v.failure_report}")

if __name__ == "__main__":
    main()
```

---

## 观测关键指标

运行时，你应该观察以下关键指标来评估 Harness 系统的效果：

```python
# harness/observability.py

class HarnessObservabilityDashboard:
    """
    Harness 可观测性仪表盘
    
    六大核心指标（参考 Harness Engineering 最佳实践）：
    """
    
    METRICS = {
        "context_utilization": "上下文窗口利用率（目标：始终 < 70%）",
        "iterations_per_task": "每任务循环次数（目标：< 10 次）",
        "validation_pass_rate": "验证一次通过率（目标：> 80%）",
        "compression_trigger_count": "上下文压缩触发次数（越少越好）",
        "tool_call_success_rate": "工具调用成功率（目标：> 95%）",
        "task_completion_latency": "任务完成时延（秒）",
    }
    
    def print_dashboard(self, agent_stats: dict) -> None:
        """打印可观测性仪表盘"""
        print("\n" + "="*50)
        print("📊 Harness 可观测性仪表盘")
        print("="*50)
        
        for metric, description in self.METRICS.items():
            value = agent_stats.get(metric, "N/A")
            print(f"  {metric}: {value}")
            print(f"  └─ {description}")
            print()
```

---

## 本节小结

你刚刚构建了一个包含以下能力的 Harness 系统：

| 组件 | 实现的功能 |
|------|-----------|
| **上下文管理器** | 实时监控利用率，40% 时自动触发渐进式压缩 |
| **工具注册表** | 权限分层、路径遍历防护、审计日志 |
| **验证门控** | 强制测试 + Lint + 防作弊检查，未通过不允许完成 |
| **死循环检测** | 监控同一文件的重复编辑，注入"换思路"建议 |
| **可观测性** | 六大核心指标追踪 |

**下一步**：尝试在你自己的项目中集成这套 Harness 系统。从最基础的验证门控开始——这一项改进通常就能将你的 Agent 的任务成功率提升 20-40%。

---

## 扩展阅读

想进一步完善你的 Harness 系统，可以考虑：

1. **加入熵治理**（第9.2节的支柱五）：每周自动扫描代码库，创建清洁 PR
2. **多 Agent 上下文隔离**（第9.2节的支柱四）：当你开始构建多 Agent 系统时
3. **AGENTS.md 一致性检查**（第9.3节）：将 `AGENTS.md` 验证加入 CI/CD 流水线

---

*下一章：[第10章 Skill System](../chapter_skill/README.md)*
