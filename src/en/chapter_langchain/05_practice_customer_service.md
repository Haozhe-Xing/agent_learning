# 12.5 Practice: Multi-Function Customer Service Agent

This example uses fixture data for FAQs, order lookups, product recommendations, and complaint drafts. Tools encapsulate the business logic, and `create_agent` selects tools based on the conversation. Drafts stay local. A real ticketing integration needs identity and permission checks and business approval rules.

![Customer service agent](../svg/chapter_langchain_05_customer_service.svg)

Set `OPENAI_API_KEY` and run the script from the repository root. Use `CUSTOMER_SERVICE_MODEL` to select a model with Chat Completions tool calling.

```bash
python -m pip install "langchain>=1,<2" "langchain-openai>=1,<2"
python examples/chapter12/customer_service.py
```

```python
{{#include ../../../examples/chapter12/customer_service.py}}
```

`InMemorySaver` and a stable `thread_id` preserve context for the lifetime of the CLI. Try “Where is ORD-12345678?” followed by “What did I order?” Use a durable checkpointer in production and verify that the caller owns the requested thread. See [Chapter 13](../chapter_langgraph/README.md).
