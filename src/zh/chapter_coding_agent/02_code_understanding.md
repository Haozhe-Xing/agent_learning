# 20.2 代码理解与分析能力

> **本节目标**：为 AI 编程助手实现代码索引、AST 分析和语义搜索能力。

---

## 代码索引

![AI 编程助手代码理解流程](../svg/chapter_coding_agent_02_code_flow.svg)

要让 Agent 理解代码，首先需要对代码库建立索引：

```python
import ast
import os
from dataclasses import dataclass, field

@dataclass
class CodeEntity:
    """代码实体（函数、类等）"""
    name: str
    entity_type: str  # function, class, method
    file_path: str
    start_line: int
    end_line: int
    docstring: str = ""
    signature: str = ""
    source: str = ""

class CodeIndexer:
    """代码索引器 —— 提取代码结构信息"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.entities: list[CodeEntity] = []
    
    def build_index(self) -> list[CodeEntity]:
        """构建整个项目的代码索引"""
        
        for root, dirs, files in os.walk(self.project_path):
            dirs[:] = [d for d in dirs 
                      if not d.startswith('.') 
                      and d not in ('__pycache__', 'venv', 'node_modules')]
            
            for f in files:
                if f.endswith('.py'):
                    filepath = os.path.join(root, f)
                    self._index_python_file(filepath)
        
        return self.entities
    
    def _index_python_file(self, filepath: str):
        """解析单个 Python 文件"""
        try:
            with open(filepath) as f:
                source = f.read()
            
            tree = ast.parse(source)
            rel_path = os.path.relpath(filepath, self.project_path)
            lines = source.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    self.entities.append(CodeEntity(
                        name=node.name,
                        entity_type="function",
                        file_path=rel_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                        signature=self._get_function_signature(node),
                        source='\n'.join(
                            lines[node.lineno-1:node.end_lineno or node.lineno]
                        )
                    ))
                
                elif isinstance(node, ast.ClassDef):
                    self.entities.append(CodeEntity(
                        name=node.name,
                        entity_type="class",
                        file_path=rel_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                        source='\n'.join(
                            lines[node.lineno-1:node.end_lineno or node.lineno]
                        )
                    ))
        except (SyntaxError, UnicodeDecodeError):
            pass  # 跳过无法解析的文件
    
    @staticmethod
    def _get_function_signature(node: ast.FunctionDef) -> str:
        """提取函数签名"""
        args = []
        for arg in node.args.args:
            annotation = ""
            if arg.annotation:
                annotation = f": {ast.unparse(arg.annotation)}"
            args.append(f"{arg.arg}{annotation}")
        
        return_type = ""
        if node.returns:
            return_type = f" -> {ast.unparse(node.returns)}"
        
        return f"def {node.name}({', '.join(args)}){return_type}"
```

---

## 语义搜索

用向量嵌入让 Agent 能通过自然语言搜索代码：

```python
class CodeSearchEngine:
    """代码语义搜索引擎"""
    
    def __init__(self, entities: list[CodeEntity], embeddings):
        self.entities = entities
        self.embeddings = embeddings
        self._vectors = None
    
    def build(self):
        """构建搜索索引"""
        # 为每个代码实体生成描述文本
        descriptions = []
        for entity in self.entities:
            desc = (
                f"{entity.entity_type} {entity.name} "
                f"in {entity.file_path}: "
                f"{entity.docstring or entity.signature}"
            )
            descriptions.append(desc)
        
        # 批量生成嵌入向量
        self._vectors = self.embeddings.embed_documents(descriptions)
    
    def search(self, query: str, top_k: int = 5) -> list[CodeEntity]:
        """语义搜索"""
        import numpy as np
        
        query_vector = self.embeddings.embed_query(query)
        
        # 计算相似度
        similarities = [
            float(np.dot(query_vector, v) / (
                np.linalg.norm(query_vector) * np.linalg.norm(v)
            ))
            for v in self._vectors
        ]
        
        # 返回 top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.entities[i] for i in top_indices]
```

---

## 代码理解 Agent

```python
class CodeUnderstandingAgent:
    """代码理解 Agent"""
    
    def __init__(self, llm, indexer: CodeIndexer, searcher: CodeSearchEngine):
        self.llm = llm
        self.indexer = indexer
        self.searcher = searcher
    
    async def explain_code(self, file_path: str) -> str:
        """解释一个文件的代码"""
        entities = [e for e in self.indexer.entities if e.file_path == file_path]
        
        code_summary = "\n".join(
            f"- {e.entity_type} `{e.name}`: {e.docstring or '无文档'}"
            for e in entities
        )
        
        prompt = f"""请用通俗易懂的语言解释这个文件的结构和功能。

文件：{file_path}
包含的代码实体：
{code_summary}

请先概述文件的整体功能，然后逐一解释每个重要的函数/类。
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def answer_question(self, question: str) -> str:
        """回答关于代码的问题"""
        
        # 搜索相关代码
        relevant = self.searcher.search(question, top_k=5)
        
        context = "\n\n".join(
            f"### {e.file_path} - {e.name}\n```python\n{e.source}\n```"
            for e in relevant
        )
        
        prompt = f"""基于以下代码片段，回答用户的问题。

{context}

问题：{question}

要求：引用具体的文件和函数名来支持你的回答。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
```

---

## 小结

| 组件 | 功能 |
|------|------|
| CodeIndexer | 解析代码结构，提取函数/类信息 |
| CodeSearchEngine | 语义搜索，通过自然语言找代码 |
| CodeUnderstandingAgent | 解释代码、回答代码相关问题 |

> **下一节预告**：理解了代码之后，我们来实现代码生成和修改能力。

---

[20.3 代码生成与修改能力](./03_code_generation.md)
