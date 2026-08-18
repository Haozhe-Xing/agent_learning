# 23.5 集成设计：从组件到助手

> **本节目标**：把 23.2–23.4 的组件按职责组装成一个 AI 编程助手，并说清它在仓库中的真实运行方式。

![完整AI编程助手组件集成架构](../svg/chapter_coding_05_full.svg)

> ⚠️ **诚实说明（可运行性）**：23.2–23.4 的 `CodeIndexer`、`CodeSearchEngine`、`CodeGenerator`、`CodeValidator`、`TestGenerator`、`BugFixer` 是在书中**以代码片段形式给出**的参考实现，依赖 `langchain_openai` 与你的 API Key，并非本仓库里可直接 `import` 的包。要让它们真正跑起来，你需要把各节代码保存为独立模块（如 `code_indexer.py` 等）并自行安装依赖——本节展示的是**组装逻辑**，不是"开箱即跑"的文件。
>
> 如果你想要一个**已经测试通过、默认离线即可运行**的最小 Agent 底座，请直接用仓库根目录的 `reference-agent/`（含工具调用、权限守卫、FastAPI 服务、评估与 MCP 示例）。第 12–23 章的"实战"均以其为准。

---

## 组件的组装逻辑

下面把各组件组合成 `AICodeAssistant`。它依赖前面几节定义的类；运行前请先把那些类整理成可导入的模块。

```python
"""
AI 编程助手（组装逻辑）
依赖：23.2 CodeIndexer / CodeSearchEngine
      23.3 CodeGenerator / CodeValidator
      23.4 TestGenerator / BugFixer
前置：将上述类保存为独立模块并 pip install langchain-openai
"""
import asyncio
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from code_indexer import CodeIndexer
from code_search import CodeSearchEngine
from code_generator import CodeGenerator
from test_generator import TestGenerator
from bug_fixer import BugFixer


class AICodeAssistant:
    """AI 编程助手组装实现"""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0)
        self.embeddings = OpenAIEmbeddings()

        self.indexer = CodeIndexer(project_path)
        entities = self.indexer.build_index()

        self.searcher = CodeSearchEngine(entities, self.embeddings)
        self.searcher.build()

        self.generator = CodeGenerator(self.llm)
        self.test_gen = TestGenerator(self.llm)
        self.bug_fixer = BugFixer(self.llm)

        print(f"已索引 {len(entities)} 个代码实体")

    async def chat(self, user_input: str) -> str:
        intent = await self._classify_intent(user_input)
        handlers = {
            "explain": self._handle_explain,
            "generate": self._handle_generate,
            "fix": self._handle_fix,
            "test": self._handle_test,
            "search": self._handle_search,
        }
        handler = handlers.get(intent, self._handle_general)
        return await handler(user_input)

    async def _classify_intent(self, user_input: str) -> str:
        prompt = (
            "判断用户意图，只回复一个词："
            "explain/generate/fix/test/search/general。\n用户说：" + user_input
        )
        return (await self.llm.ainvoke(prompt)).content.strip().lower()

    async def _handle_explain(self, query: str) -> str:
        results = self.searcher.search(query, top_k=3)
        if not results:
            return "未找到相关代码。"
        context = "\n\n".join(
            f"**{e.file_path}** - `{e.name}`\n```python\n{e.source}\n```"
            for e in results
        )
        return (await self.llm.ainvoke(
            f"用通俗语言解释以下代码：\n\n{context}\n\n用户问题：{query}"
        )).content

    async def _handle_generate(self, query: str) -> str:
        result = await self.generator.generate(query)
        return f"```python\n{result.code}\n```\n\n{result.explanation}"

    async def _handle_search(self, query: str) -> str:
        results = self.searcher.search(query, top_k=5)
        if not results:
            return "未找到相关代码。"
        return "\n".join(
            f"{i}. **{e.name}** ({e.entity_type}) @ {e.file_path}:L{e.start_line}"
            for i, e in enumerate(results, 1)
        )

    async def _handle_fix(self, query: str) -> str:
        results = self.searcher.search(query, top_k=3)
        if not results:
            return "请提供具体的错误信息和相关文件路径。"
        code = results[0].source
        fix = await self.bug_fixer.diagnose_and_fix(
            code=code, error_message=query, file_path=results[0].file_path
        )
        return f"原因：{fix.get('root_cause')}\n修复：\n```python\n{fix.get('fixed_code', code)}\n```"

    async def _handle_test(self, query: str) -> str:
        results = self.searcher.search(query, top_k=1)
        if not results:
            return "请指定要生成测试的文件或函数。"
        tests = await self.test_gen.generate_tests(
            source_code=results[0].source, file_path=results[0].file_path
        )
        return f"为 `{results[0].file_path}` 生成的测试：\n\n{tests}"

    async def _handle_general(self, query: str) -> str:
        return (await self.llm.ainvoke(
            f"你是一个专业编程助手。项目路径：{self.project_path}\n用户问题：{query}"
        )).content
```

要点：**意图分类只是路由，真正的可用性取决于前面各组件是否真的可运行**——尤其是 `CodeValidator`（语法/安全检查）和 `BugFixer` 的"修复→测试→不过再修"闭环。

---

## 自动修复闭环（设计模式）

这是 Coding Agent 最有价值的部分，也是必须"带验证"的环节。下面给出带上限的修复循环设计（伪代码/简化代码，非本仓库直接运行文件）：

```python
async def auto_fix_loop(bug_fixer, run_tests, code, test_code,
                         file_path, max_attempts: int = 3) -> dict:
    """修复 → 跑测试 → 不过就再修，直到通过或达上限"""
    current_code = code
    passed, error_msg = run_tests(current_code, test_code)
    if passed:
        return {"success": True, "attempts": 0, "code": current_code}

    for attempt in range(1, max_attempts + 1):
        fix = await bug_fixer.diagnose_and_fix(
            code=current_code, error_message=error_msg, file_path=file_path
        )
        current_code = fix["fixed_code"]
        passed, error_msg = run_tests(current_code, test_code)
        if passed:
            return {"success": True, "attempts": attempt,
                    "code": current_code, "last_fix": fix["fix_description"]}

    return {"success": False, "attempts": max_attempts,
            "code": current_code, "last_error": error_msg,
            "note": "已达最大修复次数，建议人工介入"}
```

设计要点：①每轮把**真实报错**反馈给模型，不凭空再修；②必须设 `max_attempts` 防止在两个错误版本间死循环烧 token；③修复后跑全量回归，不只看"这个 Bug 没了"；④失败优雅退出转人工。这与第 18 章成本控制、第 19 章"AI 产物必须人工把关"一致。

---

## 小结

| 功能 | 实现方式 |
|------|---------|
| 代码搜索 | 向量嵌入 + 余弦相似度 |
| 代码理解 | AST 分析 + LLM 解释 |
| 代码生成 | 结构化输出 + 质量验证 |
| 测试生成 | LLM 生成 pytest 测试 |
| Bug 修复 | 错误分析 + 代码修复 |

> 🎓 **本章总结**：我们从零构建了一个 AI 编程助手，它能理解代码、搜索代码、生成代码、写测试和修 Bug。虽然这是一个简化版本，但它展示了构建此类工具的核心思路。

---

## 📝 本章练习

读完本章，先合上书用自己的话回答下面的问题，再展开参考答案对照。

**练习 1（概念）**：23.2 节用 Python 的 `ast`（抽象语法树）来给代码建索引，提取函数名、签名、文档字符串。为什么本项目要用 AST 解析，而不是简单地"按关键字搜索代码文本"？请结合 Coding Agent 的实际需求说明 AST 方式好在哪里。

<details>
<summary>参考答案</summary>

**关键字文本搜索的局限**：如果只是把代码当成普通文本来搜（比如 `grep "def login"`），机器其实并不"理解"代码的结构。它分不清一个 `login` 是函数名、变量名、还是注释里出现的字；也拿不到这个函数从第几行到第几行、参数有哪些、返回什么类型、有没有文档说明。

**AST 解析的优势**（见 23.2）：`ast.parse` 把源码解析成"语法树"，于是 Agent 能精确地拿到结构化信息：

1. **准确识别实体类型**：知道这是 `FunctionDef`（函数）还是 `ClassDef`（类），不会把注释或字符串里的同名词当成函数。
2. **拿到精确的位置和范围**：`node.lineno` 和 `end_lineno` 给出实体的起止行，方便后续精确读取、修改或定位 Bug。
3. **提取结构化元数据**：函数签名（参数+类型注解+返回值）、docstring 都能干净地取出来，这些正是构建"代码索引 / 语义搜索描述"的高质量素材（23.2 的 `CodeSearchEngine` 就是用这些描述去生成向量的）。
4. **为可靠修改打基础**：Coding Agent 要做的不只是"读"，还要"改"。基于 AST 的精确行号，修改才能落到正确的位置，而不是误伤同名文本。

一句话：**Coding Agent 需要"理解代码结构"，而不只是"匹配字符串"，AST 正是把代码从文本升级为结构化数据的工具。** 这也是 23.1 提到的 Devin、SWE-Agent、Cursor 等产品共同采用 AST/LSP 做代码理解的原因。

</details>

**练习 2（辨析）**：23.3 节的 `CodeGenerator` 用 `with_structured_output(GeneratedCode)` 让 LLM 输出结构化结果，而且生成后还要过一遍 `CodeValidator`。有同学说："LLM 这么强，生成的代码直接用就行，验证器是多余的。"请反驳这个观点。

<details>
<summary>参考答案</summary>

这个观点忽略了 23.3 开头强调的——**生成代码比生成普通文本难得多**：代码必须语法正确、逻辑正确、风格一致、考虑边界。LLM 再强，也是"概率生成"，不能保证每次都对。验证器不是多余，而是必要的"安全网"：

1. **语法可能就是错的**：LLM 偶尔会漏个括号、缩进错位。`CodeValidator._check_python_syntax` 用 `ast.parse` 一跑就知道能不能编译，这是文本层面看不出来的。
2. **会引入安全隐患**：LLM 可能生成 `eval()`、`os.system()`、`pickle.loads()` 这类危险调用。验证器的 `_check_security` 专门拦截这些——这点和第 19 章"代码沙箱在执行前做 AST 安全检查"是一脉相承的。
3. **结构化输出 ≠ 内容正确**：`with_structured_output` 只保证返回的 JSON 字段齐全（有 code、explanation、dependencies），但不保证 code 字段里的代码真的能跑、真的安全。结构对了，内容仍要验证。
4. **生产中要"信任但要核实"**：正如 23.4 的最佳实践所说，AI 生成的代码"必须经过人工审查后再合并"。验证器是人工审查之前的第一道自动关卡，能快速筛掉明显有问题的产物，提高效率。

所以正确的流程是 **生成 → 自动验证（语法/风格/安全）→ 人工审查 → 合并**，验证器是这条质量链上不可省略的一环。

</details>

**练习 3（动手）**：本章的 Bug 修复目前是"修一次就交"。请参考 23.4 的"测试-诊断-修复闭环"思想，设计并写出一个**带验证的自动修复循环** `auto_fix_loop`：让 Agent 修复后自动跑测试，如果还没通过就把新的报错喂回去再修，最多修 N 次。用伪代码或简化代码表达核心逻辑即可。

<details>
<summary>参考答案</summary>

核心思想：把"诊断→修复→验证"做成一个**带上限的循环**，每轮把上一轮的真实报错反馈给 Agent，直到测试通过或达到最大次数。这正是 SWE-bench 类 Coding Agent 的核心工作模式。

```python
async def auto_fix_loop(
    bug_fixer,          # 23.4 的 BugFixer
    run_tests,          # 跑测试的函数：返回 (是否通过, 报错信息)
    code: str,
    test_code: str,
    file_path: str,
    max_attempts: int = 3,
) -> dict:
    """带验证的自动修复循环：修复 → 跑测试 → 不过就再修"""
    current_code = code

    # 先跑一次，确认确实有问题
    passed, error_msg = run_tests(current_code, test_code)
    if passed:
        return {"success": True, "attempts": 0, "code": current_code}

    for attempt in range(1, max_attempts + 1):
        print(f"第 {attempt} 次尝试修复，当前报错：{error_msg[:80]}")

        # 1) 让 Agent 基于"当前代码 + 当前真实报错"诊断并修复
        fix = await bug_fixer.diagnose_and_fix(
            code=current_code,
            error_message=error_msg,
            file_path=file_path,
        )
        current_code = fix["fixed_code"]

        # 2) 用修复后的代码重新跑测试（回归验证）
        passed, error_msg = run_tests(current_code, test_code)

        # 3) 通过就提前结束
        if passed:
            return {
                "success": True,
                "attempts": attempt,
                "code": current_code,
                "last_fix": fix["fix_description"],
            }

    # 修了 max_attempts 次仍没过，交给人工
    return {
        "success": False,
        "attempts": max_attempts,
        "code": current_code,
        "last_error": error_msg,
        "note": "已达最大修复次数，建议人工介入",
    }
```

**几个关键设计点**：

1. **每轮反馈真实报错**：不是凭空再修，而是把上一轮跑测试得到的 `error_msg` 喂回给 Agent，让它"看到"上次没修好——这是闭环的灵魂。
2. **必须设最大次数 `max_attempts`**：否则 Agent 可能在两个错误版本之间反复横跳，陷入死循环、烧掉大量 token（这也呼应第 18 章的成本控制和第 20 章的预算治理）。
3. **修复后一定要跑回归测试**：正如 23.4 最佳实践强调的，"自动修复后必须运行全量测试套件，防止修复引入新问题"——不能只看"这个 Bug 没了"，还要确认"没把别的搞坏"。
4. **失败要能优雅退出并转人工**：修不好不等于崩溃，而是返回明确状态交给人审查。这与本章和第 19 章反复强调的"AI 产物必须人工把关"一致。

</details>

---

[第24章 项目实战：智能数据分析 Agent](../chapter_24_data_agent/README.md)
