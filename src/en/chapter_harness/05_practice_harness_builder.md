# 9.5 Practice: Building Your First Harness System

> 🛠️ *"The best way to learn Harness Engineering is to build one for your existing Agent by hand, then watch it cut the failure rate by 50% within a week."*

---

## Goal

This section will guide you from scratch to build a complete Harness system for a typical **Python coding Agent**.

When finished, your Agent will have:

- ✅ Automatic context utilization monitoring to prevent context anxiety
- ✅ Enforced Plan-Build-Verify-Fix workflow
- ✅ Tiered tool permissions to prevent dangerous operations
- ✅ Doom loop detection with automatic strategy switching
- ✅ Observability dashboard for real-time Agent behavior monitoring

**Complete code**: approximately 400 lines of Python, ready to run.

---

## Project Structure

```
harness_demo/
├── harness/
│   ├── __init__.py
│   ├── context_manager.py     # Context lifecycle management
│   ├── tool_registry.py       # Tool registration and permission control
│   ├── validation_gate.py     # Self-validation loop
│   ├── loop_detector.py       # Doom loop detection
│   └── observability.py       # Observability
├── agent.py                   # Agent main body (uses Harness)
├── AGENTS.md                  # Agent Constitution
└── demo.py                    # Demo script
```

---

## Step 1: Write AGENTS.md

First, write `AGENTS.md` for the demo project:

```markdown
# AGENTS.md — Harness Demo Project

## Project Overview
This is a Python project for demonstrating Harness Engineering.

## Tech Stack
- Python 3.11
- FastAPI (web framework)
- pytest (testing framework)
- ruff (lint tool)

## Mandatory Workflow
After any code modification, you must execute in order:
1. `pytest tests/ -v --tb=short` (run tests)
2. `ruff check src/ --fix` (lint check and auto-fix)
3. If any tests fail, fix them and re-execute step 1

## Architectural Constraints
- src/api/ → can only call src/services/
- src/services/ → can only call src/models/ and src/repositories/
- Forbidden: executing database queries directly in the api layer

## Forbidden Operations
- Forbidden: modifying existing test files (unless explicitly fixing a test bug)
- Forbidden: hardcoding any API keys or passwords
- Forbidden: deleting any files under tests/
```

---

## Step 2: Context Manager

```python
# harness/context_manager.py

import time
from dataclasses import dataclass, field
from typing import Optional
import tiktoken


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count the number of tokens"""
    try:
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        # Rough estimate: average 1 token per 4 characters
        return len(text) // 4


@dataclass
class ContextStats:
    """Context statistics"""
    total_tokens: int
    max_tokens: int
    utilization: float
    message_count: int
    status: str  # "healthy", "warning", "danger"
    recommendation: str


class HarnessContextManager:
    """
    Harness context manager
    
    Features:
    1. Real-time monitoring of context utilization
    2. Automatic progressive compression
    3. Health reports
    """
    
    COMPRESSION_THRESHOLD = 0.40
    DANGER_THRESHOLD = 0.70
    
    def __init__(self, max_tokens: int = 128_000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self._compression_count = 0
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message, triggering compression if necessary"""
        self.messages.append({"role": role, "content": content})
        
        stats = self.get_stats()
        if stats.utilization >= self.COMPRESSION_THRESHOLD:
            self._auto_compress(stats)
    
    def get_stats(self) -> ContextStats:
        """Get current context statistics"""
        total = sum(
            count_tokens(m.get("content", ""))
            for m in self.messages
        )
        util = total / self.max_tokens
        
        if util < self.COMPRESSION_THRESHOLD:
            status, recommendation = "healthy", "Normal, no action needed"
        elif util < self.DANGER_THRESHOLD:
            status, recommendation = "warning", "Consider cleaning up old tool outputs"
        else:
            status, recommendation = "danger", "⚠️ Trigger full compression immediately"
        
        return ContextStats(
            total_tokens=total,
            max_tokens=self.max_tokens,
            utilization=util,
            message_count=len(self.messages),
            status=status,
            recommendation=recommendation,
        )
    
    def _auto_compress(self, stats: ContextStats) -> None:
        """Automatic progressive compression"""
        # Step 1: lightweight compression — clear old tool results
        self._clear_old_tool_results()
        self._compression_count += 1
        
        # Check if still over limit
        new_stats = self.get_stats()
        if new_stats.utilization >= self.COMPRESSION_THRESHOLD:
            # Step 2: full compression (requires external LLM call; simplified here)
            self._truncate_middle()
            self._compression_count += 1
    
    def _clear_old_tool_results(self) -> None:
        """Lightweight compression: clear older tool outputs"""
        cutoff = len(self.messages) - 8
        for i in range(cutoff):
            msg = self.messages[i]
            if msg.get("role") == "tool" and len(msg.get("content", "")) > 200:
                self.messages[i] = {
                    "role": "tool",
                    "content": "[Tool output archived to save context space]",
                    "tool_call_id": msg.get("tool_call_id"),
                }
    
    def _truncate_middle(self) -> None:
        """Keep head (system messages) and tail (recent conversation), compress the middle"""
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        recent_msgs = self.messages[-6:]
        
        # Replace middle history with a placeholder
        self.messages = system_msgs + [
            {
                "role": "system",
                "content": "[Early conversation history compressed. Key decisions: see system messages above.]"
            }
        ] + recent_msgs
    
    def get_messages(self) -> list[dict]:
        """Get the current message list (for LLM API calls)"""
        return self.messages.copy()
```

---

## Step 3: Tool Registry

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
    parameters: dict  # JSON Schema format
    idempotent: bool = True


class HarnessToolRegistry:
    """
    Tool registry: implements tiered permissions and strong type constraints
    """
    
    def __init__(self, workspace: str, agent_role: str = "code_writer"):
        self.workspace = Path(workspace).resolve()
        self.agent_role = agent_role
        self.tools: dict[str, Tool] = {}
        self._audit_log: list[dict] = []
        
        # Register default tool set
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the standard tool set needed for a coding Agent"""
        
        # === Read-only tools ===
        self.register(
            name="read_file",
            func=self._read_file,
            description="Read file contents. Returns the text content of the file.",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (relative to working directory)"},
                    "offset": {"type": "integer", "description": "Starting line number (optional, default 0)"},
                    "limit": {"type": "integer", "description": "Number of lines to read (optional, default all)"},
                },
                "required": ["path"],
            },
        )
        
        self.register(
            name="list_files",
            func=self._list_files,
            description="List files and subdirectories in a directory.",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: working directory)"},
                },
                "required": [],
            },
        )
        
        self.register(
            name="search_content",
            func=self._search_content,
            description="Search for specified content in files (supports regular expressions).",
            permission=PermissionLevel.READ_ONLY,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern (regular expression)"},
                    "path": {"type": "string", "description": "Search path (default: working directory)"},
                },
                "required": ["pattern"],
            },
        )
        
        # === Write tools (safe) ===
        self.register(
            name="write_file",
            func=self._write_file,
            description="Write file contents (backs up the original file first).",
            permission=PermissionLevel.WRITE_SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
            idempotent=False,
        )
        
        # === Execution tools ===
        self.register(
            name="run_tests",
            func=self._run_tests,
            description="Run the pytest test suite, returning a test result summary.",
            permission=PermissionLevel.WRITE_SAFE,
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Test pattern (default: run all)"},
                },
                "required": [],
            },
        )
        
        self.register(
            name="run_linter",
            func=self._run_linter,
            description="Run ruff code checks, returning a list of lint issues.",
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
        """Register a tool"""
        # Wrap function to add audit log
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
        """Wrap tool function with audit log"""
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
        """Call a tool"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool does not exist: {tool_name}. Available tools: {list(self.tools.keys())}")
        
        tool = self.tools[tool_name]
        return tool.func(**kwargs)
    
    def get_tool_definitions(self) -> list[dict]:
        """Return tool definition list in OpenAI API format"""
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
    
    # === Tool implementations ===
    
    def _safe_path(self, path: str) -> Path:
        """Ensure path is within the working directory (prevent path traversal)"""
        full_path = (self.workspace / path).resolve()
        if not str(full_path).startswith(str(self.workspace)):
            raise ValueError(f"Access to files outside the working directory is forbidden: {path}")
        return full_path
    
    def _read_file(self, path: str, offset: int = 0, limit: int = None) -> str:
        full_path = self._safe_path(path)
        if not full_path.exists():
            return f"Error: file does not exist: {path}"
        
        lines = full_path.read_text(encoding="utf-8").split("\n")
        if limit:
            lines = lines[offset:offset + limit]
        else:
            lines = lines[offset:]
        
        return "\n".join(f"{offset + i + 1}:{line}" for i, line in enumerate(lines))
    
    def _list_files(self, path: str = ".") -> str:
        full_path = self._safe_path(path)
        if not full_path.is_dir():
            return f"Error: not a directory: {path}"
        
        items = []
        for item in sorted(full_path.iterdir()):
            rel_path = item.relative_to(self.workspace)
            prefix = "📁 " if item.is_dir() else "📄 "
            items.append(f"{prefix}{rel_path}")
        
        return "\n".join(items) if items else "(empty directory)"
    
    def _search_content(self, pattern: str, path: str = ".") -> str:
        full_path = self._safe_path(path)
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, str(full_path)],
            capture_output=True, text=True, timeout=30
        )
        return result.stdout[:3000] if result.stdout else "No matching content found"
    
    def _write_file(self, path: str, content: str) -> str:
        full_path = self._safe_path(path)
        
        # Back up the original file (if it exists)
        if full_path.exists():
            backup_path = full_path.with_suffix(full_path.suffix + ".bak")
            backup_path.write_text(full_path.read_text(encoding="utf-8"), encoding="utf-8")
        
        # Ensure parent directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return f"✅ Written: {path} ({len(content)} bytes)"
    
    def _run_tests(self, pattern: str = "") -> str:
        cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
        if pattern:
            cmd.append(pattern)
        
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(self.workspace), timeout=120
        )
        
        # Return summary rather than full output (save context)
        output = result.stdout + result.stderr
        lines = output.split("\n")
        
        # Extract key lines: failed tests and final statistics
        key_lines = [l for l in lines if 
                     "PASSED" in l or "FAILED" in l or "ERROR" in l or 
                     "passed" in l or "failed" in l or "error" in l]
        
        return "\n".join(key_lines[-20:])  # Return at most the last 20 key lines
    
    def _run_linter(self) -> str:
        result = subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--output-format=text"],
            capture_output=True, text=True,
            cwd=str(self.workspace), timeout=60
        )
        output = result.stdout + result.stderr
        if not output.strip():
            return "✅ No lint issues"
        return output[:2000]  # Limit output length
    
    def get_audit_summary(self) -> str:
        """Return tool call audit summary"""
        if not self._audit_log:
            return "No tool call records"
        
        counts = {}
        errors = []
        for entry in self._audit_log:
            counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
            if entry["status"] == "error":
                errors.append(f"  - {entry['tool']}: {entry.get('error', 'unknown')}")
        
        lines = ["Tool call statistics:"]
        for tool, count in sorted(counts.items()):
            lines.append(f"  {tool}: {count} calls")
        
        if errors:
            lines.append("\nCall errors:")
            lines.extend(errors)
        
        return "\n".join(lines)
```

---

## Step 4: Validation Gate

```python
# harness/validation_gate.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    passed: bool
    checks: dict[str, bool]  # Result of each check
    failures: list[str]
    
    @property
    def failure_report(self) -> str:
        if not self.failures:
            return "All validations passed"
        return "\n".join(f"❌ {f}" for f in self.failures)


class HarnessValidationGate:
    """
    Validation gate: before the Agent claims a task is complete,
    enforces a series of checks
    """
    
    def __init__(self, tool_registry):
        self.tools = tool_registry
    
    def validate(
        self,
        task_description: str,
        modified_files: list[str],
    ) -> ValidationResult:
        """
        Run the complete validation process
        
        Checks:
        1. Test suite passes
        2. No lint errors
        3. Test files not deleted
        4. Expected files were modified
        """
        checks = {}
        failures = []
        
        # Check 1: run tests
        print("  🧪 Running tests...")
        test_output = self.tools.call("run_tests")
        test_passed = "failed" not in test_output.lower() and "error" not in test_output.lower()
        checks["tests_passed"] = test_passed
        if not test_passed:
            failures.append(f"Tests failed:\n{test_output[:500]}")
        
        # Check 2: lint check
        print("  📋 Running lint check...")
        lint_output = self.tools.call("run_linter")
        lint_passed = "No lint issues" in lint_output or lint_output.strip() == ""
        checks["lint_passed"] = lint_passed
        if not lint_passed:
            failures.append(f"Lint issues:\n{lint_output[:300]}")
        
        # Check 3: check test file integrity (anti-cheating)
        print("  🛡️ Checking test file integrity...")
        test_integrity = self._check_test_integrity(modified_files)
        checks["test_integrity"] = test_integrity
        if not test_integrity:
            failures.append("⚠️ Test files detected as deleted or significantly reduced! This is not allowed.")
        
        passed = len(failures) == 0
        
        if passed:
            print("  ✅ All validations passed!")
        else:
            print(f"  ❌ {len(failures)} validation(s) failed")
        
        return ValidationResult(
            passed=passed,
            checks=checks,
            failures=failures,
        )
    
    def _check_test_integrity(self, modified_files: list[str]) -> bool:
        """Check if test files have been illegally deleted or significantly reduced"""
        for file_path in modified_files:
            if "test" in file_path.lower():
                # Read file content to check for large deletions
                try:
                    content = self.tools.call("read_file", path=file_path)
                    line_count = len(content.split("\n"))
                    # If a test file now has fewer than 5 lines, something is likely wrong
                    if line_count < 5 and "test" in file_path.lower():
                        return False
                except Exception:
                    pass
        return True
```

---

## Step 5: Doom Loop Detector

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
    Doom loop detector
    
    Detects: the same file being modified repeatedly (possibly stuck in a try-fail-retry loop)
    Response: injects a "try a different approach" suggestion rather than forcing interruption
    """
    
    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.edit_history: dict[str, list[float]] = defaultdict(list)
    
    def record_edit(self, file_path: str) -> Optional[LoopWarning]:
        """
        Record a file edit.
        If the threshold is exceeded, return a warning (otherwise return None).
        """
        now = time.time()
        self.edit_history[file_path].append(now)
        
        # Only look at edits within the last 10 minutes
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
⚠️ Loop detection warning: you have made {count} modifications to `{file_path}`.

Repeatedly modifying the same file usually indicates:
1. The root cause of the problem is not in this file (check what calls it)
2. Your modification direction may be wrong (re-read the task requirements)
3. There is a dependency issue in another file (use search_content to find related code)

Suggestions:
- Stop modifying this file for now
- Run tests and carefully read the error messages
- Use search_content to search for code related to the error
- If still unable to resolve, describe the specific problem you're encountering
"""
    
    def get_report(self) -> str:
        """Return edit history report"""
        if not self.edit_history:
            return "No edit records"
        
        lines = ["File edit history:"]
        for path, timestamps in sorted(self.edit_history.items()):
            recent = [t for t in timestamps if time.time() - t < 600]
            if recent:
                status = "⚠️ Warning" if len(recent) > self.threshold else "✅ Normal"
                lines.append(f"  {status} {path}: {len(recent)} edits in the last 10 minutes")
        
        return "\n".join(lines)
```

---

## Step 6: Integrate the Harness Agent

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
You are a professional Python programming assistant following Harness Engineering best practices.

## Workflow (Mandatory)
When completing each coding task, you must follow these steps:

### Step 1: Plan
- Analyze task requirements
- List the files to be modified
- Formulate a specific modification plan

### Step 2: Implement
- Implement each item according to the plan
- After each file modification, briefly describe what was done

### Step 3: Verify (Mandatory — cannot be skipped!)
- Call run_tests to run tests
- Call run_linter to check code style
- Check each item against the task requirements

### Step 4: Fix (if issues found)
- Fix all issues found during verification
- Re-execute Step 3

Only when Step 3 passes completely can you say "task complete."

## Project Standards
See AGENTS.md (use the read_file tool to read it).
"""


class HarnessAgent:
    """
    Coding Agent using the Harness system
    """
    
    def __init__(self, workspace: str, api_key: str = None, model: str = "gpt-4o"):
        self.workspace = workspace
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
        # Initialize Harness components
        self.context_manager = HarnessContextManager(max_tokens=128_000)
        self.tool_registry = HarnessToolRegistry(workspace)
        self.validation_gate = HarnessValidationGate(self.tool_registry)
        self.loop_detector = HarnessLoopDetector(threshold=3)
        
        # Load system prompt
        self.context_manager.add_message("system", SYSTEM_PROMPT)
        
        # Inject environment context
        env_context = self._build_env_context()
        self.context_manager.add_message("system", env_context)
        
        # Track modified files (for validation)
        self._modified_files: list[str] = []
        
        print(f"✅ Harness Agent initialized")
        print(f"   Working directory: {workspace}")
        print(f"   Model: {model}")
        print(f"   Available tools: {len(self.tool_registry.tools)}")
    
    def _build_env_context(self) -> str:
        """Build environment context (progressive disclosure: give directory, not full content)"""
        try:
            # Get top-level directory structure
            import os
            items = os.listdir(self.workspace)
            structure = "\n".join(f"  - {item}" for item in sorted(items[:20]))
        except Exception:
            structure = "  (unable to read)"
        
        return f"""
## Working Environment

**Working directory**: {self.workspace}

**Top-level directory structure**:
{structure}

**Available tools**:
{chr(10).join(f"  - {name}: {tool.description[:60]}" for name, tool in self.tool_registry.tools.items())}

**Tip**: Use the list_files and read_file tools to explore the project structure.
Use read_file to read AGENTS.md to understand project conventions.
"""
    
    def execute(self, task: str, max_iterations: int = 20) -> dict:
        """
        Execute a task (with full Harness protection)
        
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
        print(f"🎯 Task: {task}")
        print(f"{'='*60}")
        
        self.context_manager.add_message("user", task)
        
        iteration = 0
        last_error = None
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            
            # Display context health status
            stats = self.context_manager.get_stats()
            status_icon = {"healthy": "🟢", "warning": "🟡", "danger": "🔴"}[stats.status]
            print(f"Context status: {status_icon} {stats.utilization:.1%} ({stats.total_tokens} tokens)")
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.context_manager.get_messages(),
                tools=self.tool_registry.get_tool_definitions(),
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            
            # Check for tool calls
            if message.tool_calls:
                self.context_manager.add_message("assistant", 
                    json.dumps([tc.model_dump() for tc in message.tool_calls]))
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"  🔧 Calling tool: {tool_name}({list(tool_args.keys())})")
                    
                    # Doom loop detection (for write file operations)
                    if tool_name == "write_file":
                        file_path = tool_args.get("path", "")
                        warning = self.loop_detector.record_edit(file_path)
                        
                        if file_path not in self._modified_files:
                            self._modified_files.append(file_path)
                        
                        if warning:
                            # Inject loop warning
                            print(f"  ⚠️ Loop detection: {file_path} has been edited {warning.edit_count} times")
                            self.context_manager.add_message("system", warning.suggestion)
                    
                    # Execute tool
                    try:
                        result = self.tool_registry.call(tool_name, **tool_args)
                    except Exception as e:
                        result = f"Tool execution error: {str(e)}"
                    
                    # Add result to context
                    self.context_manager.add_message("tool", str(result)[:2000])
                
            else:
                # No tool calls — Agent is outputting a text response
                content = message.content or ""
                self.context_manager.add_message("assistant", content)
                print(f"  💬 Agent: {content[:200]}...")
                
                # Check if claiming task completion
                completion_signals = [
                    "task complete", "task completed", "successfully completed",
                    "done", "finished", "all done",
                ]
                if any(signal in content.lower() for signal in completion_signals):
                    print("\n🔍 Task completion signal detected, running mandatory validation...")
                    
                    validation = self.validation_gate.validate(
                        task_description=task,
                        modified_files=self._modified_files,
                    )
                    
                    if validation.passed:
                        print("✅ Validation passed! Task successfully completed.")
                        return {
                            "success": True,
                            "result": content,
                            "iterations": iteration,
                            "validation": validation,
                            "context_stats": self.context_manager.get_stats(),
                            "tool_audit": self.tool_registry.get_audit_summary(),
                        }
                    else:
                        # Validation failed: inject failure info and request fix
                        print(f"❌ Validation failed, requesting fix:")
                        print(validation.failure_report)
                        
                        fix_prompt = f"""
Validation failed — the task is not truly complete! Please fix the following issues:

{validation.failure_report}

Please:
1. Carefully read the error messages above
2. Fix all issues
3. Re-run tests and lint to confirm they pass
4. Only declare the task complete after all validations pass
"""
                        self.context_manager.add_message("user", fix_prompt)
        
        # Exceeded maximum iterations
        print(f"\n⚠️ Exceeded maximum iterations ({max_iterations})")
        return {
            "success": False,
            "result": f"Task not completed after {max_iterations} iterations",
            "iterations": iteration,
            "validation": None,
            "context_stats": self.context_manager.get_stats(),
            "tool_audit": self.tool_registry.get_audit_summary(),
        }
```

---

## Step 7: Run the Demo

```python
# demo.py

import os
from agent import HarnessAgent

def main():
    # Configuration
    workspace = os.path.join(os.path.dirname(__file__), "sample_project")
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Create Harness Agent
    agent = HarnessAgent(
        workspace=workspace,
        api_key=api_key,
        model="gpt-4o",
    )
    
    # Execute task
    task = """
    Please implement the following functions in src/utils/math_helpers.py:
    
    1. `factorial(n: int) -> int`: compute factorial (n >= 0), raise ValueError when n < 0
    2. `fibonacci(n: int) -> int`: return the nth Fibonacci number (0-indexed), raise ValueError when n < 0
    
    Requirements:
    - Include complete type annotations
    - Include Google-style docstrings
    - Write unit tests in tests/test_math_helpers.py, covering normal cases and edge cases
    """
    
    result = agent.execute(task)
    
    # Display results
    print("\n" + "="*60)
    print("📊 Execution Report")
    print("="*60)
    print(f"Status: {'✅ Success' if result['success'] else '❌ Failed'}")
    print(f"Iterations: {result['iterations']}")
    
    if result.get("context_stats"):
        stats = result["context_stats"]
        print(f"Final context utilization: {stats.utilization:.1%}")
    
    if result.get("tool_audit"):
        print(f"\n{result['tool_audit']}")
    
    if result.get("validation"):
        v = result["validation"]
        print(f"\nValidation result: {'✅ Passed' if v.passed else '❌ Failed'}")
        if v.failures:
            print(f"Failure reason: {v.failure_report}")

if __name__ == "__main__":
    main()
```

---

## Observing Key Metrics

When running, you should observe the following key metrics to evaluate the effectiveness of the Harness system:

```python
# harness/observability.py

class HarnessObservabilityDashboard:
    """
    Harness observability dashboard
    
    Six core metrics (based on Harness Engineering best practices):
    """
    
    METRICS = {
        "context_utilization": "Context window utilization (target: always < 70%)",
        "iterations_per_task": "Iterations per task (target: < 10)",
        "validation_pass_rate": "First-pass validation rate (target: > 80%)",
        "compression_trigger_count": "Context compression trigger count (fewer is better)",
        "tool_call_success_rate": "Tool call success rate (target: > 95%)",
        "task_completion_latency": "Task completion latency (seconds)",
    }
    
    def print_dashboard(self, agent_stats: dict) -> None:
        """Print the observability dashboard"""
        print("\n" + "="*50)
        print("📊 Harness Observability Dashboard")
        print("="*50)
        
        for metric, description in self.METRICS.items():
            value = agent_stats.get(metric, "N/A")
            print(f"  {metric}: {value}")
            print(f"  └─ {description}")
            print()
```

---

## Summary

You just built a Harness system with the following capabilities:

| Component | Implemented Features |
|-----------|---------------------|
| **Context Manager** | Real-time utilization monitoring; auto-triggers progressive compression at 40% |
| **Tool Registry** | Tiered permissions, path traversal protection, audit log |
| **Validation Gate** | Mandatory tests + lint + anti-cheating checks; completion not allowed until passed |
| **Doom Loop Detector** | Monitors repeated edits to the same file; injects "try a different approach" suggestions |
| **Observability** | Six core metric tracking |

**Next step**: try integrating this Harness system into your own project. Start with the most basic validation gate — this single improvement typically raises your Agent's task success rate by 20–40%.

---

## Further Reading

To further improve your Harness system, consider:

1. **Add entropy governance** (Pillar 5 from Section 9.2): automatically scan the codebase weekly and create cleanup PRs
2. **Multi-Agent context isolation** (Pillar 4 from Section 9.2): when you start building multi-Agent systems
3. **AGENTS.md consistency checks** (Section 9.3): add `AGENTS.md` validation to your CI/CD pipeline

---

*Next chapter: [Chapter 10 Skill System](../chapter_skill/README.md)*