# Part IV: Collaboration and Integration

Bring a single-Agent task, permission boundary and acceptance baseline. This part addresses two separate questions: whether roles should share the work, and how tools or other systems cross a protocol boundary. You need not adopt both.

| Chapter | Trigger | Artifact | Comparison and failure case |
|---|---|---|---|
| [16 Multi-Agent Collaboration](./chapter_18_multi_agent/README.md) | Independently verifiable subtasks or meaningful role boundaries | Handoff contract, shared budget and aggregate acceptance | Compare against one Agent on identical tasks and total budget; failed subtasks cannot become a delivered result |
| [17 Protocols](./chapter_19_protocol/README.md) | Tools cross process or system boundaries | Request/response contract, errors and connection lifecycle | A valid request still works after a malformed one; connectivity grants no business permission |

A single Agent can need MCP. A state graph does not automatically need multiple Agents. Without a collaboration or integration requirement, proceed directly to [Part V](./part5.md).
