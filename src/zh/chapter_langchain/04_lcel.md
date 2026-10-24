# 12.4 LCEL：LangChain 表达式语言

LCEL（LangChain Expression Language）是 LangChain 的核心构建语言，用 `|` 符号将组件连接成处理管道。如果你用过 Unix 管道（`cat file.txt | grep "error" | wc -l`），LCEL 的思路完全相同：数据从左到右流过一系列处理组件，每个组件接收上一个的输出作为自己的输入。

LCEL 的核心抽象是 **Runnable 协议**——所有组件都实现了统一接口（`invoke`、`stream`、`batch` 等），因此任何组件都可以自由组合。你写好一条链之后，自动获得了流式输出、异步调用和批处理能力，不需要额外编码。

![LCEL 管道架构：数据流与组合模式](../svg/chapter_langchain_04_lcel.svg)

## 四个最常用原语

| 原语 | 作用 | 使用场景 |
|------|------|---------|
| `\|` 操作符 | 把两个 Runnable 串联（背后是 `__or__` 方法） | 所有链的基础 |
| `RunnablePassthrough` | 原样传递输入，常用于保留原始数据 | 同时需要"检索上下文"和"原问题"时 |
| `RunnableLambda` | 把普通 Python 函数包装成 Runnable | 在管道中插入自定义逻辑 |
| `RunnableParallel` | 并行执行多个 Runnable，结果合并为字典 | RAG 同时检索 + 传问题；多路分析 |

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4.1-mini")

# 基础链：提示 → 模型 → 解析
chain = ChatPromptTemplate.from_messages([("human", "{question}")]) | llm | StrOutputParser()

# RunnableParallel：同时保留"检索到的上下文"和"原问题"
rag_chain = (
    RunnableParallel({"context": lambda x: retrieve(x["question"]),
                      "question": lambda x: x["question"]})
    | ChatPromptTemplate.from_messages([("system", "基于上下文回答：{context}"),
                                        ("human", "{question}")])
    | llm | StrOutputParser()
)

# RunnableLambda：把普通函数嵌入管道
json_chain = (ChatPromptTemplate.from_messages([("system", "转JSON，含title和priority"),
                                                ("human", "{description}")])
              | llm | StrOutputParser()
              | RunnableLambda(lambda text: json.loads(text[text.find("{"):text.rfind("}")+1])))
```

> 💡 **直觉理解**：`RunnablePassthrough` 像是管道中的"三通接头"——它不改变水流，只是把主流复制一份送到旁路（比如把原问题也发给模型）。`RunnableLambda` 则是你往流水线上"加一个手工工位"的标准接口。把原语组合好，复杂的 RAG、多路分析都能用声明式表达。

## 错误处理与重试：生产级韧性

生产环境中，LLM API 可能因网络抖动、速率限制偶尔失败。LCEL 内置两种恢复机制：

- **`with_retry`**：自动重试，支持指数退避（避免限流时雪崩）
- **`with_fallbacks`**：主链失败后切到备用链（如主模型 GPT-4.1 → 备用 gpt-4.1-mini），保证可用性

```python
resilient = (ChatPromptTemplate.from_messages([("human", "{input}")])
             | llm.with_retry(stop_after_attempt=3, wait_exponential_jitter=True)
             | StrOutputParser())

fallback = (ChatPromptTemplate.from_messages([("human", "{input}")])
            | ChatOpenAI(model="gpt-4.1-mini") | StrOutputParser())
chain_with_fallback = resilient.with_fallbacks([fallback])
```

## 链的组合与复用

子链可以像积木一样复用：定义一次 `summarize_chain`，就能在"先摘要再翻译""先摘要再分类"等多处复用。

```python
summarize = ChatPromptTemplate.from_messages([("system", "压缩为50字摘要"), ("human", "{text}")]) | llm | StrOutputParser()
translate = ChatPromptTemplate.from_messages([("system", "翻译成英文"), ("human", "{text}")]) | llm | StrOutputParser()

summarize_then_translate = summarize | RunnableLambda(lambda x: {"text": x}) | translate
```

---

## 小结

LCEL 的核心优势：
- **统一接口**：所有组件都是 Runnable，支持相同调用方式
- **声明式**：代码即文档，清晰表达数据流向
- **内置支持**：自动支持流式、异步、批处理
- **可组合性**：子链自由组合复用

> 📌 LCEL 解决了"线性管道"的组合问题。但**循环、条件分支、持久状态、人工介入**这些 Agent 真正复杂的需求，依然超出 LCEL 的表达力——这正是 LangGraph 的舞台（第 13 章）。

---

*下一节：[12.5 实战：多功能客服 Agent](./05_practice_customer_service.md)*
