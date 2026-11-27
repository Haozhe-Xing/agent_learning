# 12.5 实战：多功能客服 Agent

客服实战用固定数据演示 FAQ、订单查询、商品推荐和投诉草稿。工具函数封装业务逻辑，`create_agent` 根据会话选择工具。示例只生成投诉草稿。真实工单写入时，应用应验证身份、授权和业务审批。

![多功能客服 Agent 系统架构](../svg/chapter_langchain_05_customer_service.svg)

在仓库根目录设置 `OPENAI_API_KEY`，然后运行脚本。可通过 `CUSTOMER_SERVICE_MODEL` 指定支持 Chat Completions 工具调用的模型。

```bash
python -m pip install "langchain>=1,<2" "langchain-openai>=1,<2"
python examples/chapter12/customer_service.py
```

```python
{{#include ../../../examples/chapter12/customer_service.py}}
```

`InMemorySaver` 搭配固定的 `thread_id` 保留该 CLI 会话的上下文，退出后状态消失。先问 `Where is ORD-12345678?`，再问 `What did I order?`。生产环境需要持久化 checkpointer，并在服务端验证调用者与会话的归属关系。参见[第 13 章](../chapter_langgraph/README.md)。
