# 12.2 Chain：构建处理管道

Chain（链）是 LangChain 的核心概念——将多个处理步骤串联成可复用的流水线。你可以把 Chain 想象成一条装配线：原材料（用户输入）从一端进入，经过多个加工站（提示模板、LLM、解析器等），最终从另一端输出成品（结构化结果）。

在 LangChain 中，Chain 用 **LCEL（LangChain Expression Language）** 构建。LCEL 的核心符号是 `|`（管道符），工作方式类似 Unix 命令行管道——前一个组件的输出自动成为下一个组件的输入。本节先建立"四种 Chain 模式"的心智模型，再给一个最小可运行示例，最后给出"何时应选哪种模式"的决策表。

![LCEL四种Chain模式](../svg/chapter_langchain_02_chain_patterns.svg)

## 四种 Chain 模式的心智模型

| 模式 | 数据流向 | 典型场景 | 核心原语 |
|------|---------|---------|---------|
| **顺序链** | A → B → C，前步输出喂后步 | 提取关键词 → 生成摘要 | `\|` 串联 |
| **并行链** | 同一输入同时跑多条链，结果汇入字典 | 情感分析 + 摘要 + 关键词同时做 | `RunnableParallel` |
| **条件链** | 按输入特征路由到不同分支 | 客服按意图（技术/业务/投诉）分流 | `RunnableBranch` |
| **流式链** | 逐字输出，不等待完整生成 | 类 ChatGPT 打字机效果 | `.stream()` / `.astream()` |

> 💡 **直觉理解**：这四种模式对应人类处理任务的四种本能——**按顺序做**（先理解再总结）、**同时做**（边看边听边想）、**看情况做**（问题不同走不同流程）、**边做边说**（不必等全部想完才开口）。LCEL 只是把这四种本能变成了可组合的算子。

## 最小可运行示例

下面一段代码同时覆盖了"顺序 + 并行 + 流式"三种模式，是后续所有 Chain 的基础：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough

llm = ChatOpenAI(model="gpt-4.1-mini")

# 顺序链：提示 → LLM → 解析
translate_chain = ChatPromptTemplate.from_messages([
    ("system", "你是一位专业翻译，将文本翻译成{target_lang}"),
    ("human", "{text}")
]) | llm | StrOutputParser()

# 并行链：同一输入同时跑多个独立分析，结果汇入字典
def analyze(text: str) -> dict:
    sentiment = (ChatPromptTemplate.from_messages([
        ("system", "只返回：正面/负面/中性"), ("human", "{text}")])
        | llm | StrOutputParser()).invoke({"text": text})
    summary = (ChatPromptTemplate.from_messages([
        ("system", "用一句话概括"), ("human", "{text}")])
        | llm | StrOutputParser()).invoke({"text": text})
    return {"情感": sentiment, "摘要": summary}

# 流式链：逐字返回
async def stream(text: str):
    chain = ChatPromptTemplate.from_messages([("system", "你是有帮助的助手"), ("human", "{q}")]) | llm | StrOutputParser()
    async for chunk in chain.astream({"q": text}):
        print(chunk, end="", flush=True)
```

## 条件链的关键陷阱

`RunnableBranch` 按条件函数路由到不同分支。一个常见错误是**在条件函数里也调用 LLM**——如果每次判断都先跑一次分类模型，条件分支一多，token 消耗会翻倍。正确做法是"先分类一次并缓存结果，再路由"：

```python
from langchain_core.runnables import RunnableLambda, RunnableBranch

# 先用一个 Lambda 把意图分类一次，挂到状态字典上
branch = (
    RunnableLambda(lambda x: {**x, "_intent": classify_intent(x)})
    | RunnableBranch(
        (lambda x: "技术问题" in x["_intent"], tech_chain),
        (lambda x: "投诉" in x["_intent"], complaint_chain),
        default_chain,  # 兜底分支
    )
)
```

> ⚠️ **性能要点**：条件路由的判断逻辑应尽量轻量（关键词、规则、或一次性的分类结果缓存），避免每个分支都重复触发 LLM 调用。

## 何时选用哪种模式（决策表）

| 你的需求 | 选哪种模式 | 理由 |
|---------|-----------|------|
| 步骤固定、前步结果喂后步 | 顺序链 | 最自然，直接 `\|` 串联 |
| 多个独立分析彼此无关 | 并行链 | 节省延迟，结果天然汇入 |
| 路径由运行时输入决定 | 条件链 | 用 `RunnableBranch` 分流 |
| 用户不能等完整结果 | 流式链 | `.stream()` 默认支持 |
| 既要并行又要顺序 | 组合 | `RunnableParallel` 内嵌顺序链 |

## 流式输出：为什么"边做边说"很重要

实际应用中，用户不希望等 LLM 生成完整回复才看到内容。LCEL 构建的所有链都**天然支持流式输出**，无需额外代码：

```python
# 任何 LCEL 链都能 .stream() / .astream()
for chunk in chain.stream({"question": "什么是量子纠缠？"}):
    print(chunk, end="", flush=True)
```

---

## 小结

LCEL（`|` 管道语法）是 LangChain 的核心构建方式：
- **顺序链**：步骤间传递结果
- **并行链**：`RunnableParallel` 同时执行
- **条件链**：`RunnableBranch` 按条件路由（注意缓存分类结果）
- **流式输出**：所有 LCEL 链都支持 `.stream()` 和 `.astream()`

> 📌 记住：Chain 解决的是"线性、可预测"的流程。当你的需求出现**循环、回溯、人工介入**时，链就不够用了——这正是第 13 章 LangGraph 要解决的。

---

*下一节：[12.3 使用 LangChain 构建 Agent](./03_langchain_agents.md)*
