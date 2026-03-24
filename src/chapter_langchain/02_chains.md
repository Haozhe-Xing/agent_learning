# Chain：构建处理管道

Chain（链）是 LangChain 的核心概念——将多个处理步骤串联成可复用的流水线。你可以把 Chain 想象成一条装配线：原材料（用户输入）从一端进入，经过多个加工站（提示模板、LLM、解析器等），最终从另一端输出成品（结构化结果）。

在 LangChain 中，Chain 使用 **LCEL（LangChain Expression Language）** 语法来构建。LCEL 的核心符号是 `|`（管道符），它的工作方式类似于 Unix 命令行中的管道——前一个组件的输出会自动成为下一个组件的输入。

本节将通过四种常见的 Chain 模式，带你掌握 LCEL 的核心用法。

![LCEL四种Chain模式](../svg/chapter_langchain_02_chain_patterns.svg)

## LCEL：现代链式语法

### 基础准备

首先导入必要的模块。LCEL 的核心构建块包括：`ChatPromptTemplate`（提示模板）、`ChatOpenAI`（LLM）、`StrOutputParser`（字符串解析器）以及各类 `Runnable` 组件。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from operator import itemgetter

llm = ChatOpenAI(model="gpt-4o-mini")

# ============================
# 基础链：提示 → LLM → 解析
# ============================

# 翻译链
translate_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业翻译，将文本翻译成{target_lang}"),
    ("human", "{text}")
])

translate_chain = translate_prompt | llm | StrOutputParser()

result = translate_chain.invoke({
    "target_lang": "英语",
    "text": "人工智能正在改变世界"
})
print(result)

# ============================
# 顺序链：一步步处理
# ============================

# 顺序链将多个步骤串联起来——前一步的输出成为后一步的输入。
# 这里的示例是"先提取关键词，再结合原文生成摘要"的两步流程。
# 关键挑战：后续步骤需要同时访问前一步的结果和原始输入。
# 解决方案：用 RunnableParallel 同时执行提取和传递原始输入。
keyword_prompt = ChatPromptTemplate.from_messages([
    ("system", "从文本中提取3个关键词，用逗号分隔，只输出关键词"),
    ("human", "{text}")
])

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下关键词和原文，生成一段简洁摘要"),
    ("human", "关键词：{keywords}\n原文：{original_text}")
])

# 方式1：使用 RunnablePassthrough 传递原始输入
analysis_chain = (
    RunnableParallel(
        keywords=keyword_prompt | llm | StrOutputParser(),
        original_text=RunnablePassthrough()
    )
    | RunnableLambda(lambda x: {
        "keywords": x["keywords"],
        "original_text": x["original_text"]["text"]
    })
    | summary_prompt | llm | StrOutputParser()
)

result = analysis_chain.invoke({"text": "Python是一种强大的编程语言，广泛用于AI开发"})
print(result)

# ============================
# 并行链：同时执行多个任务
# ============================

# RunnableParallel 可以让多个链同时运行，适合需要对同一输入
# 执行多种独立分析的场景。每个分支独立执行，互不影响，
# 最终结果汇聚成一个字典返回。这比逐个串行调用快得多。

def analyze_text_parallel(text: str) -> dict:
    """同时进行情感分析和摘要生成"""
    
    sentiment_prompt = ChatPromptTemplate.from_messages([
        ("system", "对文本做情感分析，只返回：正面/负面/中性"),
        ("human", "{text}")
    ])
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "用一句话概括文本主要内容"),
        ("human", "{text}")
    ])
    
    keywords_prompt = ChatPromptTemplate.from_messages([
        ("system", "提取文本的5个关键词，用逗号分隔"),
        ("human", "{text}")
    ])
    
    # 并行执行
    parallel_chain = RunnableParallel(
        sentiment=sentiment_prompt | llm | StrOutputParser(),
        summary=summary_prompt | llm | StrOutputParser(),
        keywords=keywords_prompt | llm | StrOutputParser()
    )
    
    return parallel_chain.invoke({"text": text})

result = analyze_text_parallel("今天发布的新版本修复了很多bug，性能也大幅提升！")
print(f"情感：{result['sentiment']}")
print(f"摘要：{result['summary']}")
print(f"关键词：{result['keywords']}")

# ============================
# 条件链：根据条件路由
# ============================

# 条件链（RunnableBranch）根据输入的特征选择不同的处理分支。
# 典型场景：客服系统根据用户意图（技术问题、业务咨询、投诉等）
# 路由到不同的专业处理链。
#
# ⚠️ 性能要点：RunnableBranch 会依次检查每个条件函数，
# 如果条件函数中调用了 LLM，应该先执行一次分类、缓存结果，
# 避免每个条件分支都重复调用 LLM（见下方优化后的实现）。

from langchain_core.runnables import RunnableBranch

def classify_intent(input_dict: dict) -> str:
    """分类用户意图"""
    text = input_dict.get("text", "")
    
    classify_prompt = ChatPromptTemplate.from_messages([
        ("system", "判断以下文本的意图类型，只返回：技术问题/业务咨询/投诉/其他"),
        ("human", "{text}")
    ])
    
    intent = (classify_prompt | llm | StrOutputParser()).invoke({"text": text})
    return intent.strip()

# 不同意图对应不同处理链
tech_chain = ChatPromptTemplate.from_messages([
    ("system", "你是技术支持工程师，提供详细技术解答"),
    ("human", "{text}")
]) | llm | StrOutputParser()

business_chain = ChatPromptTemplate.from_messages([
    ("system", "你是业务顾问，提供专业业务建议"),
    ("human", "{text}")
]) | llm | StrOutputParser()

complaint_chain = ChatPromptTemplate.from_messages([
    ("system", "你是客户关系经理，处理投诉时要有耐心和同理心"),
    ("human", "{text}")
]) | llm | StrOutputParser()

default_chain = ChatPromptTemplate.from_messages([
    ("system", "你是通用客服助手"),
    ("human", "{text}")
]) | llm | StrOutputParser()

# 路由链
# 注意：先用 RunnableLambda 分类一次并缓存结果到字典中，避免多次调用 LLM
branch_chain = (
    RunnableLambda(lambda x: {**x, "_intent": classify_intent(x)})
    | RunnableBranch(
        (lambda x: "技术问题" in x["_intent"], tech_chain),
        (lambda x: "业务咨询" in x["_intent"], business_chain),
        (lambda x: "投诉" in x["_intent"], complaint_chain),
        default_chain  # 默认分支
    )
)

# 测试
response = branch_chain.invoke({"text": "API返回500错误怎么办？"})
print(response)
```

## 流式输出链

在实际应用中，用户不希望等待 LLM 生成完整回复后才看到内容。流式输出可以让回复逐字显示（类似 ChatGPT 的打字效果），大幅提升用户体验。LCEL 构建的所有链都天然支持流式输出，无需额外修改代码。

```python
# LCEL 天然支持流式输出
async def stream_response(question: str):
    """流式输出"""
    chain = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的助手"),
        ("human", "{question}")
    ]) | llm | StrOutputParser()
    
    print("回答：", end="", flush=True)
    async for chunk in chain.astream({"question": question}):
        print(chunk, end="", flush=True)
    print()  # 换行

import asyncio
asyncio.run(stream_response("什么是量子纠缠？"))
```

---

## 小结

LCEL（`|` 管道语法）是 LangChain 的核心构建方式：
- **顺序链**：步骤间传递结果
- **并行链**：`RunnableParallel` 同时执行
- **条件链**：`RunnableBranch` 按条件路由
- **流式输出**：所有 LCEL 链都支持 `.stream()` 和 `.astream()`

---

*下一节：[11.3 使用 LangChain 构建 Agent](./03_langchain_agents.md)*
