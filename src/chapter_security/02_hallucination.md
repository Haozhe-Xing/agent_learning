# 幻觉问题与事实性保障

> **本节目标**：理解 LLM 幻觉的成因，掌握减少幻觉、提高事实性的实用技术。

---

## 什么是幻觉（Hallucination）？

![LLM 幻觉类型与防御策略](../svg/chapter_security_02_hallucination_types.svg)

幻觉是指 LLM 生成的内容看起来流畅合理，但实际上是错误的或编造的。就像一个"知识渊博"的朋友，当他不知道答案时，不会说"我不知道"，而是自信满满地编一个听起来很对的答案。

### 幻觉的类型

| 类型 | 描述 | 示例 |
|------|------|------|
| 事实性幻觉 | 生成了与事实不符的信息 | "Python 是 1995 年发布的"（实际是 1991 年） |
| 引用幻觉 | 引用了不存在的论文或链接 | "根据 Smith et al. (2023) 的研究..."（论文不存在） |
| 逻辑幻觉 | 推理过程看似合理但实际有错 | 数学计算正确展示过程但结果错误 |
| 指令幻觉 | 声称执行了某操作但实际没有 | "我已经帮你发送了邮件"（实际没有发送功能） |

---

## 减少幻觉的策略

### 策略 1：要求引用来源

强制 Agent 标注信息来源，无来源则声明不确定：

```python
ANTI_HALLUCINATION_PROMPT = """
## 事实性要求（必须遵守）

1. **有依据才说**：只提供你确信正确的信息
2. **标注来源**：涉及具体事实、数据时，标注来源
3. **承认不知道**：不确定的内容，明确说"我不确定，建议您查证"
4. **区分事实与观点**：事实用陈述句，观点用"我认为"、"通常来说"

### 示例
✅ "Python 3.12 于 2023 年 10 月发布，引入了更好的错误提示。"
✅ "关于这个问题，我不太确定具体的数值，建议您参考官方文档。"
❌ "这个库的最新版本是 5.2.1。"（如果你不确定版本号）
"""
```

### 策略 2：RAG 事实核查

用检索到的文档来验证 Agent 的回答：

```python
class FactChecker:
    """基于 RAG 的事实核查器"""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def check(self, claim: str) -> dict:
        """核查一个声明的事实性"""
        
        # 1. 检索相关文档
        docs = self.retriever.invoke(claim)
        
        if not docs:
            return {
                "verdict": "unverifiable",
                "confidence": 0.0,
                "explanation": "未找到相关文档来验证此声明"
            }
        
        # 2. 让 LLM 判断
        context = "\n\n".join(doc.page_content for doc in docs[:3])
        
        check_prompt = f"""基于以下参考文档，判断这个声明是否正确。

声明：{claim}

参考文档：
{context}

请回复 JSON 格式：
{{
    "verdict": "supported" 或 "contradicted" 或 "unverifiable",
    "confidence": 0.0-1.0,
    "explanation": "判断依据"
}}"""
        
        response = self.llm.invoke(check_prompt)
        import json
        return json.loads(response.content)
    
    def check_response(self, response: str) -> list[dict]:
        """核查整个回答中的关键声明"""
        
        # 先提取关键声明
        extract_prompt = f"""从以下回答中提取所有可验证的事实性声明。
每个声明一行，只提取客观事实，不要提取观点。

回答：
{response}

声明列表："""
        
        claims_response = self.llm.invoke(extract_prompt)
        claims = [
            c.strip().lstrip("- ·•")
            for c in claims_response.content.strip().split("\n")
            if c.strip()
        ]
        
        # 逐一核查
        results = []
        for claim in claims:
            result = self.check(claim)
            result["claim"] = claim
            results.append(result)
        
        return results
```

### 策略 3：自我一致性检查

让 Agent 多次回答同一个问题，如果答案不一致，说明可能存在幻觉：

```python
async def self_consistency_check(
    question: str,
    llm,
    num_samples: int = 3,
    temperature: float = 0.7
) -> dict:
    """自我一致性检查：多次生成并比较"""
    import asyncio
    
    # 生成多个回答
    tasks = []
    for _ in range(num_samples):
        tasks.append(llm.ainvoke(
            question,
            temperature=temperature
        ))
    
    responses = await asyncio.gather(*tasks)
    answers = [r.content for r in responses]
    
    # 让 LLM 判断一致性
    consistency_prompt = f"""以下是同一个问题的 {num_samples} 个回答。
请判断它们是否一致。

问题：{question}

""" + "\n\n".join(
        f"回答 {i+1}：{a}" for i, a in enumerate(answers)
    ) + """

请回复：
1. 一致性评分（0-1）
2. 最可靠的回答是哪个
3. 不一致的具体方面
"""
    
    analysis = await llm.ainvoke(consistency_prompt)
    
    return {
        "answers": answers,
        "analysis": analysis.content,
        "num_samples": num_samples
    }
```

### 策略 4：工具兜底

对于需要精确数据的场景，强制使用工具而非记忆：

```python
def create_grounded_agent(llm, tools):
    """创建一个'接地'的 Agent —— 优先使用工具获取事实"""
    
    system_prompt = """你是一个严谨的助手。

## 核心原则：工具优先
- 涉及具体数据（价格、日期、数量等），必须使用工具查询
- 涉及实时信息（天气、新闻、股价等），必须使用工具查询
- 只有在工具无法获取时，才使用你的知识回答
- 使用知识回答时，必须标注"根据我的知识"

## 禁止行为
- 不要编造具体的数字、日期、链接
- 不要假装执行了操作（如"已发送邮件"）
- 不要引用不确定的来源
"""
    
    # 使用 LangChain 构建 Agent
    from langchain.agents import create_openai_tools_agent, AgentExecutor  # legacy，新项目推荐 LangGraph
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)
```

---

## 小结

| 策略 | 作用 | 适用场景 |
|------|------|---------|
| 要求引用来源 | 迫使模型为信息找依据 | 知识问答 |
| RAG 事实核查 | 用检索文档验证回答 | 文档问答 |
| 自我一致性 | 通过多次生成检测不确定性 | 高风险决策 |
| 工具兜底 | 用工具获取精确数据 | 数据查询 |

> 📖 **想深入了解幻觉检测与缓解的学术前沿？** 请阅读 [14.6 论文解读：安全与可靠性前沿研究](./06_paper_readings.md)，涵盖 FActScore、SelfCheckGPT、Self-Consistency、CoVe 等核心论文的深度解读。

> **下一节预告**：Agent 不仅要"说得对"，还要"做得安全"——权限控制至关重要。

---

[下一节：14.3 权限控制与沙箱隔离 →](./03_permission_sandbox.md)
