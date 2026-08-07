# 21.2 Code Understanding and Analysis

> **Section Goal**: Implement code indexing, AST analysis, and semantic search capabilities for the AI coding assistant.

---

## Code Indexing

![AI Coding Assistant Code Understanding Flow](../svg/chapter_coding_agent_02_code_flow.svg)

To enable an Agent to understand code, we first need to build an index of the codebase:

```python
import ast
import os
from dataclasses import dataclass, field

@dataclass
class CodeEntity:
    """Code entity (function, class, etc.)"""
    name: str
    entity_type: str  # function, class, method
    file_path: str
    start_line: int
    end_line: int
    docstring: str = ""
    signature: str = ""
    source: str = ""

class CodeIndexer:
    """Code indexer — extracts code structure information"""
    
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.entities: list[CodeEntity] = []
    
    def build_index(self) -> list[CodeEntity]:
        """Build a code index for the entire project"""
        
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
        """Parse a single Python file"""
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
            pass  # Skip files that cannot be parsed
    
    @staticmethod
    def _get_function_signature(node: ast.FunctionDef) -> str:
        """Extract function signature"""
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

## Semantic Search

Use vector embeddings to enable the Agent to search code using natural language:

```python
class CodeSearchEngine:
    """Code semantic search engine"""
    
    def __init__(self, entities: list[CodeEntity], embeddings):
        self.entities = entities
        self.embeddings = embeddings
        self._vectors = None
    
    def build(self):
        """Build search index"""
        # Generate description text for each code entity
        descriptions = []
        for entity in self.entities:
            desc = (
                f"{entity.entity_type} {entity.name} "
                f"in {entity.file_path}: "
                f"{entity.docstring or entity.signature}"
            )
            descriptions.append(desc)
        
        # Batch generate embedding vectors
        self._vectors = self.embeddings.embed_documents(descriptions)
    
    def search(self, query: str, top_k: int = 5) -> list[CodeEntity]:
        """Semantic search"""
        import numpy as np
        
        query_vector = self.embeddings.embed_query(query)
        
        # Calculate similarity
        similarities = [
            float(np.dot(query_vector, v) / (
                np.linalg.norm(query_vector) * np.linalg.norm(v)
            ))
            for v in self._vectors
        ]
        
        # Return top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.entities[i] for i in top_indices]
```

---

## Code Understanding Agent

```python
class CodeUnderstandingAgent:
    """Code Understanding Agent"""
    
    def __init__(self, llm, indexer: CodeIndexer, searcher: CodeSearchEngine):
        self.llm = llm
        self.indexer = indexer
        self.searcher = searcher
    
    async def explain_code(self, file_path: str) -> str:
        """Explain the code in a file"""
        entities = [e for e in self.indexer.entities if e.file_path == file_path]
        
        code_summary = "\n".join(
            f"- {e.entity_type} `{e.name}`: {e.docstring or 'No documentation'}"
            for e in entities
        )
        
        prompt = f"""Please explain the structure and functionality of this file in plain language.

File: {file_path}
Code entities:
{code_summary}

Please start with an overview of the file's overall functionality, then explain each important function/class.
"""
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    async def answer_question(self, question: str) -> str:
        """Answer questions about code"""
        
        # Search for relevant code
        relevant = self.searcher.search(question, top_k=5)
        
        context = "\n\n".join(
            f"### {e.file_path} - {e.name}\n```python\n{e.source}\n```"
            for e in relevant
        )
        
        prompt = f"""Based on the following code snippets, answer the user's question.

{context}

Question: {question}

Requirement: Reference specific file and function names to support your answer."""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
```

---

## Summary

| Component | Function |
|-----------|---------|
| CodeIndexer | Parse code structure, extract function/class information |
| CodeSearchEngine | Semantic search, find code using natural language |
| CodeUnderstandingAgent | Explain code, answer code-related questions |

> **Next Section Preview**: After understanding code, let's implement code generation and modification capabilities.

---

[Next: 19.3 Code Generation and Modification →](./03_code_generation.md)
