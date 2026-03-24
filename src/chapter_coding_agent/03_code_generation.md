# 代码生成与修改能力

> **本节目标**：实现 Agent 的代码生成和修改功能，确保生成的代码质量可控。

![结构化代码生成与质量控制流程](../svg/chapter_coding_03_generation.svg)

---

## 代码生成的挑战

生成代码比生成普通文本难得多：
- 必须语法正确（一个括号不匹配就报错）
- 必须逻辑正确（不能只是"看起来对"）
- 必须与现有代码风格一致
- 必须考虑边界情况

---

## 结构化代码生成

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class GeneratedCode(BaseModel):
    """结构化的代码生成结果"""
    code: str = Field(description="生成的代码")
    language: str = Field(description="编程语言")
    explanation: str = Field(description="代码解释")
    dependencies: list[str] = Field(
        default_factory=list, description="需要安装的依赖"
    )
    usage_example: str = Field(
        default="", description="使用示例"
    )

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm.with_structured_output(GeneratedCode)
    
    async def generate(
        self,
        requirement: str,
        language: str = "python",
        style_guide: str = None
    ) -> GeneratedCode:
        """根据需求生成代码"""
        
        style_section = ""
        if style_guide:
            style_section = f"\n代码风格要求：\n{style_guide}\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一个专业的 {language} 开发者。
请根据需求生成高质量的代码。

要求：
1. 代码必须完整可运行
2. 包含必要的错误处理
3. 添加清晰的注释和文档字符串
4. 遵循 {language} 的最佳实践
{style_section}"""),
            ("human", "{requirement}")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({"requirement": requirement})
        
        return result

# 使用示例
async def demo():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    generator = CodeGenerator(llm)
    
    result = await generator.generate(
        requirement="实现一个带过期时间的 LRU 缓存",
        language="python"
    )
    
    print(result.code)
    print(f"\n依赖: {result.dependencies}")
    print(f"\n说明: {result.explanation}")
```

---

## 代码修改（Diff 模式）

修改现有代码比从头生成更复杂——需要理解上下文并精确地做出修改：

```python
class CodeModifier:
    """代码修改器 —— 基于 diff 的精确修改"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def modify(
        self,
        original_code: str,
        modification_request: str,
        file_path: str
    ) -> dict:
        """修改代码"""
        
        prompt = f"""请根据修改需求，对以下代码进行修改。

文件：{file_path}

原始代码：
```
{original_code}
```

修改需求：{modification_request}

请返回以下格式（JSON）：
{{
    "modified_code": "完整的修改后代码",
    "changes": [
        {{
            "description": "改动说明",
            "line_range": "第 X-Y 行"
        }}
    ],
    "explanation": "整体修改说明"
}}

注意：
1. 只修改必要的部分，保持其他代码不变
2. 保持原有的代码风格
3. 确保修改后的代码能正确运行
"""
        
        response = await self.llm.ainvoke(prompt)
        import json
        return json.loads(response.content)
    
    async def add_feature(
        self,
        existing_code: str,
        feature_description: str
    ) -> str:
        """在现有代码中添加新功能"""
        
        prompt = f"""在以下现有代码中添加新功能，保持与现有代码风格一致。

现有代码：
```python
{existing_code}
```

新功能需求：{feature_description}

要求：
1. 新代码自然融入现有代码结构
2. 不修改现有功能的逻辑
3. 添加必要的导入语句
4. 添加文档字符串和注释

请直接返回完整的修改后代码。
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
```

---

## 代码质量验证

生成代码后，自动验证质量：

```python
import ast
import subprocess
import tempfile

class CodeValidator:
    """代码质量验证器"""
    
    def validate(self, code: str, language: str = "python") -> dict:
        """验证代码质量"""
        results = {
            "syntax_valid": False,
            "style_issues": [],
            "security_issues": [],
            "overall_pass": False
        }
        
        if language == "python":
            results["syntax_valid"] = self._check_python_syntax(code)
            results["style_issues"] = self._check_style(code)
            results["security_issues"] = self._check_security(code)
        
        results["overall_pass"] = (
            results["syntax_valid"]
            and len(results["security_issues"]) == 0
        )
        
        return results
    
    def _check_python_syntax(self, code: str) -> bool:
        """检查 Python 语法"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _check_style(self, code: str) -> list[str]:
        """检查代码风格"""
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"第 {i} 行超过 120 字符")
            if line.rstrip() != line:
                issues.append(f"第 {i} 行有尾部空白")
        
        return issues
    
    def _check_security(self, code: str) -> list[str]:
        """检查安全问题"""
        issues = []
        
        dangerous = {
            "eval(": "使用了 eval()，可能存在代码注入风险",
            "exec(": "使用了 exec()，可能存在代码注入风险",
            "os.system(": "使用了 os.system()，建议使用 subprocess",
            "pickle.loads(": "使用了 pickle.loads()，可能存在反序列化攻击风险",
        }
        
        for pattern, warning in dangerous.items():
            if pattern in code:
                issues.append(warning)
        
        return issues
```

---

## 小结

| 组件 | 功能 |
|------|------|
| CodeGenerator | 根据需求生成完整代码 |
| CodeModifier | 精确修改现有代码 |
| CodeValidator | 语法/风格/安全自动验证 |

> **下一节预告**：代码写好了，还需要测试。让 Agent 自动生成测试并修复 Bug。

---

[下一节：19.4 测试生成与 Bug 修复 →](./04_testing_debugging.md)
