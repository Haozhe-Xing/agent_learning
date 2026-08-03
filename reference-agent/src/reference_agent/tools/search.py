from __future__ import annotations

from .base import Tool


class SearchTool(Tool):
    """A tiny built-in knowledge base used for offline demos.

    In production you would back this with a real retriever (vector DB, web
    search, internal API). The interface is identical.
    """

    name = "search"
    description = "Search a small built-in knowledge base for a keyword."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "search keyword"}},
        "required": ["query"],
    }

    KB = {
        "agent": "Agent 是能感知环境、进行推理并采取行动以达成目标的系统。",
        "rag": "RAG（检索增强生成）通过检索外部知识来提升生成的事实性。",
        "mcp": "MCP（Model Context Protocol）是 Agent 连接工具与数据源的标准协议。",
        "tool": "工具调用（Tool Use）让模型能执行代码、查询 API 等外部动作。",
    }

    def run(self, query: str) -> str:
        q = query.lower()
        for key, value in self.KB.items():
            if key in q:
                return value
        return "未找到相关结果。"
