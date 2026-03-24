# 完整项目实现

> **本节目标**：将前面几节的组件整合，构建一个可交互的 AI 编程助手。

![完整AI编程助手组件集成架构](../svg/chapter_coding_05_full.svg)

---

## 整合所有组件

```python
"""
完整的 AI 编程助手
整合：代码索引 + 语义搜索 + 代码生成 + 测试生成 + Bug 修复
"""
import asyncio
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 导入前面实现的组件（实际项目中从模块导入）
# 各组件的完整实现请参考对应章节：
# from code_indexer import CodeIndexer         # → 19.2 节
# from code_search import CodeSearchEngine     # → 19.2 节
# from code_generator import CodeGenerator     # → 19.3 节
# from test_generator import TestGenerator     # → 19.4 节
# from bug_fixer import BugFixer               # → 19.4 节
# 提示：运行本节代码前，需先将 19.2-19.4 节的代码保存为独立模块

class AICodeAssistant:
    """AI 编程助手 —— 完整实现"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.embeddings = OpenAIEmbeddings()
        
        # 初始化各组件
        self.indexer = CodeIndexer(project_path)
        entities = self.indexer.build_index()
        
        self.searcher = CodeSearchEngine(entities, self.embeddings)
        self.searcher.build()
        
        self.generator = CodeGenerator(self.llm)
        self.test_gen = TestGenerator(self.llm)
        self.bug_fixer = BugFixer(self.llm)
        
        print(f"✅ 已索引 {len(entities)} 个代码实体")
    
    async def chat(self, user_input: str) -> str:
        """处理用户输入"""
        
        # 识别意图
        intent = await self._classify_intent(user_input)
        
        if intent == "explain":
            return await self._handle_explain(user_input)
        elif intent == "generate":
            result = await self.generator.generate(user_input)
            return f"```python\n{result.code}\n```\n\n{result.explanation}"
        elif intent == "fix":
            return await self._handle_fix(user_input)
        elif intent == "test":
            return await self._handle_test(user_input)
        elif intent == "search":
            return await self._handle_search(user_input)
        else:
            return await self._handle_general(user_input)
    
    async def _classify_intent(self, user_input: str) -> str:
        """分类用户意图"""
        prompt = f"""判断用户意图，只回复一个词：
- explain: 解释代码
- generate: 生成新代码
- fix: 修复 Bug
- test: 生成测试
- search: 搜索代码
- general: 其他问题

用户说：{user_input}"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content.strip().lower()
    
    async def _handle_explain(self, query: str) -> str:
        """处理代码解释请求"""
        results = self.searcher.search(query, top_k=3)
        
        if not results:
            return "未找到相关代码。"
        
        context = "\n\n".join(
            f"**{e.file_path}** - `{e.name}`\n```python\n{e.source}\n```"
            for e in results
        )
        
        prompt = f"用通俗语言解释以下代码：\n\n{context}\n\n用户问题：{query}"
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def _handle_search(self, query: str) -> str:
        """处理代码搜索请求"""
        results = self.searcher.search(query, top_k=5)
        
        output = "🔍 搜索结果：\n\n"
        for i, entity in enumerate(results, 1):
            output += (
                f"{i}. **{entity.name}** ({entity.entity_type})\n"
                f"   📄 {entity.file_path}:L{entity.start_line}\n"
                f"   📝 {entity.docstring[:100] if entity.docstring else '无文档'}\n\n"
            )
        
        return output
    
    async def _handle_fix(self, query: str) -> str:
        """处理 Bug 修复请求"""
        # 搜索可能相关的代码
        results = self.searcher.search(query, top_k=3)
        
        if results:
            code = results[0].source
            fix = await self.bug_fixer.diagnose_and_fix(
                code=code,
                error_message=query,
                file_path=results[0].file_path
            )
            return (
                f"🔍 **原因**：{fix.get('root_cause', '未知')}\n\n"
                f"🔧 **修复**：{fix.get('fix_description', '')}\n\n"
                f"```python\n{fix.get('fixed_code', code)}\n```"
            )
        
        return "请提供具体的错误信息和相关文件路径。"
    
    async def _handle_test(self, query: str) -> str:
        """处理测试生成请求"""
        results = self.searcher.search(query, top_k=1)
        
        if results:
            entity = results[0]
            tests = await self.test_gen.generate_tests(
                source_code=entity.source,
                file_path=entity.file_path
            )
            return f"为 `{entity.file_path}` 生成的测试：\n\n{tests}"
        
        return "请指定要生成测试的文件或函数。"
    
    async def _handle_general(self, query: str) -> str:
        """处理通用问题"""
        prompt = f"""你是一个专业的编程助手。当前项目路径：{self.project_path}
        
用户问题：{query}

请尽量结合编程知识来回答。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content


async def main():
    """交互式主循环"""
    import sys
    
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    
    print("🤖 AI 编程助手")
    print(f"📁 项目: {os.path.abspath(project_path)}")
    print("=" * 50)
    print("输入 'quit' 退出\n")
    
    assistant = AICodeAssistant(project_path)
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q'):
            print("👋 再见！")
            break
        
        if not user_input:
            continue
        
        response = await assistant.chat(user_input)
        print(f"\n🤖: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 运行效果示例

```
🤖 AI 编程助手
📁 项目: /home/user/my-project
==================================================
✅ 已索引 156 个代码实体
输入 'quit' 退出

你: 搜索所有处理用户认证的函数
🤖: 🔍 搜索结果：
1. **authenticate_user** (function)
   📄 auth/service.py:L23
   📝 验证用户凭据并返回 JWT token

2. **verify_token** (function)
   📄 auth/middleware.py:L15
   📝 验证请求中的 JWT token

你: 解释一下 authenticate_user 的逻辑
🤖: `authenticate_user` 函数执行以下步骤...

你: 为 verify_token 生成测试
🤖: 为 `auth/middleware.py` 生成的测试：
    ...pytest 测试代码...
```

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

[下一章：第20章 项目实战：智能数据分析 Agent →](../chapter_data_agent/README.md)
