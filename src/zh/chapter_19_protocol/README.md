# 第17章 Agent 通信协议

**阅读层次**：工程进阶；外部工具集成按需学习。

## 本章迭代

- **上一版的不足**：宿主内的工具调用已可验收，但换成独立进程或外部服务后，请求格式、连接状态与错误行为可能不一致。
- **本章增加**：把已有工具契约映射为协议请求与响应，明确请求 ID、参数、错误和会话生命周期；业务授权仍由工具服务检查。
- **怎样验证**：已有 `reference-agent/tests/test_mcp_server.py` 在同一 stdio 进程中发送坏 JSON、非法工具名与参数，再调用合法计算并得到 96。A2A/ANP 和业务级远程权限需按正文另做读者实验，不能由最小 JSON-RPC 示例推定。
- **下一步**：将本地调用和协议调用的成功/失败记录一起交给[第18章评估](../chapter_20_evaluation/README.md)。不需要对接其他系统时保留本地接口即可。

> 🔌 *"Agent 之间需要标准化的通信方式，就像人类需要共同语言一样。"*

---

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 深入理解 MCP（Model Context Protocol）的设计与使用，能实现 Server 和 Client
- ✅ 了解 A2A（Agent-to-Agent）和 ANP（Agent Network Protocol）协议
- ✅ 掌握 Agent 间消息传递和状态共享的实现方式
- ✅ 完成基于 MCP 的工具集成实战项目

## 📚 本章结构

| 小节 |
|---|
| [17.1 MCP（Model Context Protocol）详解](./01_mcp_protocol.md) |
| [17.2 A2A（Agent-to-Agent）协议](./02_a2a_protocol.md) |
| [17.3 ANP（Agent Network Protocol）协议](./03_anp_protocol.md) |
| [17.4 Agent 间的消息传递与状态共享](./04_message_passing.md) |
| [17.5 实战：基于 MCP 的工具集成](./05_practice_mcp_integration.md) |
| [17.6 框架补充：AutoGen（多 Agent 对话框架）](./06_autogen.md) |

## 🔗 学习路径

[工具调用](../chapter_tools/README.md)与[Harness](../chapter_harness/README.md)中的任务、权限和验收记录。生产与领域交付前需通过[评估](../chapter_20_evaluation/README.md)和[安全](../chapter_21_security/README.md)检查。

将本地调用和协议调用的成功/失败记录一起交给[第18章评估](../chapter_20_evaluation/README.md)。不需要对接其他系统时保留本地接口即可。

---

*下一节：[17.1 MCP（Model Context Protocol）详解](./01_mcp_protocol.md)*
