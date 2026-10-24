# Chapter 17: Agent Communication Protocols

**Reading level**: Engineering extension; study for external integrations.

## This chapter's iteration

- **Previous limitation**: Local tool calls pass, but moving them to another process or service can change request formats, connection state and error behavior.
- **This chapter adds**: Map the existing tool contract to protocol requests and responses, including IDs, arguments, errors and session ownership. The tool service still enforces business authorization.
- **How to verify**: Supplied `reference-agent/tests/test_mcp_server.py` sends malformed JSON and invalid fields in one stdio process, then obtains 96 from a valid calculator call. A2A/ANP and remote business authorization need separate reader experiments; the small JSON-RPC server does not validate them.
- **Next step**: Carry local and protocol success/failure records into [chapter 18](../chapter_20_evaluation/README.md). Keep local interfaces when no integration boundary is needed.

> 🔌 *"Agents need standardized communication methods, just as humans need a common language."*

---

## Chapter Overview

As the Agent ecosystem rapidly evolves, standardized communication protocols are becoming increasingly important. MCP (Model Context Protocol) defines the connection standard between Agents and tools/data sources, while the A2A (Agent-to-Agent) protocol standardizes how Agents interact with each other. This chapter provides an in-depth look at the design philosophy and practical applications of these protocols.

## Chapter Goals

After completing this chapter, you will be able to:

- ✅ Understand the architectural design and core concepts of the MCP protocol
- ✅ Implement MCP Servers and Clients
- ✅ Understand the design philosophy of the A2A protocol
- ✅ Master practical patterns for message passing and state sharing between Agents
- ✅ Build a tool integration system based on MCP

## Chapter Structure

| Section |
|---|
| [17.1 MCP (Model Context Protocol) Explained](./01_mcp_protocol.md) |
| [17.2 A2A (Agent-to-Agent) Protocol](./02_a2a_protocol.md) |
| [17.3 ANP (Agent Network Protocol)](./03_anp_protocol.md) |
| [17.4 Message Passing and State Sharing Between Agents](./04_message_passing.md) |
| [17.5 Practice: Tool Integration Based on MCP](./05_practice_mcp_integration.md) |
| [17.6 Framework Supplement: AutoGen (Multi-Agent Dialogue Framework)](./06_autogen.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Carry local and protocol success/failure records into [chapter 18](../chapter_20_evaluation/README.md). Keep local interfaces when no integration boundary is needed.

---

*Next section: [17.1 MCP (Model Context Protocol) Deep Dive](./01_mcp_protocol.md)*
