# 项目架构设计

> **本节目标**：设计一个 AI 编程助手的整体架构，明确各模块的职责和交互方式。

---

## 项目目标

我们要构建一个 AI 编程助手，它能够：
- 理解代码仓库的结构和逻辑
- 回答关于代码的问题
- 生成新代码和修改现有代码
- 自动生成测试和修复 Bug

这不是一个简单的"代码补全"工具，而是一个能理解项目上下文、进行多步推理的 Agent。

![AI 编程助手 Agent 架构](../svg/chapter_coding_agent_01_architecture.svg)

---

## 整体架构

```python
"""
AI 编程助手架构

┌─────────────────────────────────────────────┐
│                用户接口层                     │
│   CLI / IDE 插件 / Web 界面                  │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              Agent 核心层                    │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│   │ 意图理解 │→ │ 任务规划 │→ │ 执行引擎 │  │
│   └──────────┘  └──────────┘  └──────────┘  │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              工具层                          │
│   代码搜索 │ 文件读写 │ AST解析 │ 测试运行  │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│              知识层                          │
│   代码索引 │ 向量数据库 │ 项目上下文         │
└─────────────────────────────────────────────┘
"""
```

---

## 核心组件设计

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class TaskType(Enum):
    """编程助手支持的任务类型"""
    CODE_QA = "code_qa"              # 代码问答
    CODE_GENERATE = "code_generate"  # 代码生成
    CODE_MODIFY = "code_modify"      # 代码修改
    CODE_REVIEW = "code_review"      # 代码审查
    TEST_GENERATE = "test_generate"  # 测试生成
    BUG_FIX = "bug_fix"            # Bug 修复
    EXPLAIN = "explain"            # 代码解释

@dataclass
class ProjectContext:
    """项目上下文信息"""
    root_path: str
    language: str
    framework: Optional[str] = None
    file_tree: list[str] = field(default_factory=list)
    dependencies: dict = field(default_factory=dict)
    
@dataclass
class CodingTask:
    """编程任务"""
    task_type: TaskType
    description: str
    target_files: list[str] = field(default_factory=list)
    context: Optional[ProjectContext] = None
    constraints: list[str] = field(default_factory=list)

class CodingAssistant:
    """AI 编程助手主类"""
    
    def __init__(self, llm, project_path: str):
        self.llm = llm
        self.project_path = project_path
        self.context = self._build_context()
    
    def _build_context(self) -> ProjectContext:
        """构建项目上下文"""
        import os
        
        # 扫描文件树
        file_tree = []
        for root, dirs, files in os.walk(self.project_path):
            # 跳过隐藏目录和常见的忽略目录
            dirs[:] = [d for d in dirs 
                      if not d.startswith('.') 
                      and d not in ('node_modules', '__pycache__', 'venv')]
            
            for f in files:
                rel_path = os.path.relpath(
                    os.path.join(root, f), self.project_path
                )
                file_tree.append(rel_path)
        
        # 检测语言和框架
        language = self._detect_language(file_tree)
        framework = self._detect_framework(file_tree)
        
        return ProjectContext(
            root_path=self.project_path,
            language=language,
            framework=framework,
            file_tree=file_tree[:200]  # 限制数量
        )
    
    def _detect_language(self, files: list[str]) -> str:
        """检测项目主要语言"""
        ext_count = {}
        lang_map = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.go': 'go', '.rs': 'rust',
        }
        
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in lang_map:
                ext_count[ext] = ext_count.get(ext, 0) + 1
        
        if ext_count:
            most_common = max(ext_count, key=ext_count.get)
            return lang_map.get(most_common, "unknown")
        return "unknown"
    
    def _detect_framework(self, files: list[str]) -> str:
        """检测使用的框架"""
        indicators = {
            "requirements.txt": "python",
            "package.json": "node",
            "Cargo.toml": "rust",
            "go.mod": "go",
        }
        
        for f in files:
            if os.path.basename(f) in indicators:
                return indicators[os.path.basename(f)]
        return None
    
    async def handle(self, task: CodingTask) -> str:
        """处理编程任务"""
        task.context = self.context
        
        handlers = {
            TaskType.CODE_QA: self._handle_qa,
            TaskType.CODE_GENERATE: self._handle_generate,
            TaskType.CODE_REVIEW: self._handle_review,
            TaskType.BUG_FIX: self._handle_bug_fix,
            TaskType.EXPLAIN: self._handle_explain,
        }
        
        handler = handlers.get(task.task_type)
        if handler:
            return await handler(task)
        else:
            return f"暂不支持任务类型: {task.task_type.value}"
    
    async def _handle_qa(self, task: CodingTask) -> str:
        """处理代码问答"""
        # 搜索相关代码 → 构建上下文 → LLM 回答
        relevant_code = self._search_relevant_code(task.description)
        
        prompt = f"""基于以下项目代码，回答用户的问题。

项目语言：{task.context.language}
相关代码：
{relevant_code}

问题：{task.description}
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _handle_generate(self, task):
        """处理代码生成"""
        prompt = f"""请根据以下需求生成代码。

项目语言：{task.context.language}
需求：{task.description}
约束：{', '.join(task.constraints) if task.constraints else '无'}

请生成完整的、可运行的代码。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _handle_review(self, task):
        """处理代码审查"""
        code = self._read_files(task.target_files)
        
        prompt = f"""请审查以下代码，指出：
1. 潜在的 Bug
2. 性能问题
3. 安全隐患
4. 代码风格问题
5. 改进建议

代码：
{code}"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _handle_bug_fix(self, task):
        """处理 Bug 修复"""
        code = self._read_files(task.target_files)
        
        prompt = f"""以下代码存在 Bug，请找出并修复。

Bug 描述：{task.description}

代码：
{code}

请提供修复后的完整代码和修复说明。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _handle_explain(self, task):
        """处理代码解释"""
        code = self._read_files(task.target_files)
        
        prompt = f"""请用通俗易懂的语言解释以下代码：

{code}

要求：
1. 先概述整体功能
2. 逐个解释重要的函数/类
3. 说明关键的设计模式或算法"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    def _search_relevant_code(self, query: str) -> str:
        """搜索相关代码（简化版，实际应使用向量搜索）"""
        # 简单的关键词匹配搜索
        import os
        
        results = []
        keywords = query.lower().split()
        
        for filepath in self.context.file_tree[:50]:
            full_path = os.path.join(self.project_path, filepath)
            try:
                with open(full_path, 'r') as f:
                    content = f.read()
                    if any(kw in content.lower() for kw in keywords):
                        results.append(f"--- {filepath} ---\n{content[:500]}")
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        return "\n\n".join(results[:5]) if results else "未找到相关代码"
    
    def _read_files(self, file_paths: list[str]) -> str:
        """读取文件内容"""
        import os
        
        contents = []
        for path in file_paths:
            full_path = os.path.join(self.project_path, path)
            try:
                with open(full_path) as f:
                    contents.append(f"--- {path} ---\n{f.read()}")
            except FileNotFoundError:
                contents.append(f"--- {path} ---\n[文件不存在]")
        
        return "\n\n".join(contents)
```

---

## 小结

| 组件 | 职责 |
|------|------|
| 用户接口层 | 接收用户指令，展示结果 |
| Agent 核心层 | 意图理解、任务规划、执行控制 |
| 工具层 | 代码搜索、文件操作、AST 分析 |
| 知识层 | 项目索引、向量检索 |

> **下一节预告**：有了架构，我们来实现代码理解能力——让 Agent 真正"读懂"代码。

---

[下一节：16.2 代码理解与分析能力 →](./02_code_understanding.md)
