# 12.7 LangChain Ecosystem 2026

> This section covers the roles of LangChain v1, Deep Agents, and managed execution, and how to migrate older tool-calling code.

---

## The LangChain Ecosystem at a Glance

LangChain has grown from a single framework into a complete ecosystem. As of 2026, the core members of the LangChain ecosystem are:

| Tool | Positioning | Core value |
|------|-------------|------------|
| **LangChain** | Core orchestration framework | Component abstractions + the LCEL expression language |
| **LangGraph** | Stateful Agent framework | Graph-based orchestration, loops, human-in-the-loop |
| **Deep Agents** | Reusable Agent Harness | Filesystem, context management, subagents, and optional planning and Skills |
| **Managed Deep Agents** | Managed delivery of the Harness | Declare Agent behavior and let LangSmith host execution |
| **LangSmith Agent Server** | Managed runtime service | Deployment, scaling, persistence |
| **LangSmith** | Agent engineering platform | Tracing, evaluation, prompt management, and deployment |

> 💡 **Evolution logic**: LangChain provides components, LangGraph supports stateful orchestration and execution, and Deep Agents combines ready-made Harness capabilities. You can run the Agent yourself or choose a managed path such as Managed Deep Agents; LangSmith covers tracing, evaluation, and deployment. Understanding these responsibilities matters more than memorizing product names.

---

## Deep Agents and Managed Deep Agents: From Harness to Managed Execution

Chapters 7–9 covered context engineering, Harnesses, and Skills separately. A research assistant, for example, may need to save findings, manage growing context, and delegate some work to subagents alongside its search tool. **Deep Agents combines these capabilities into a reusable Harness**. It builds on LangChain's Agent components and uses the LangGraph runtime for durable execution, streaming, and human-in-the-loop.[5]

### What Does Managed Execution Add?

**Managed Deep Agents (MDA) reuses the open-source Deep Agents Harness** and runs it on LangSmith Agent Server. Developers define the model, instructions, tools, and business behavior; the platform assembles the Harness and runtime infrastructure. The change is in engineering responsibilities: developers define Agent behavior, while the platform manages its runtime lifecycle.[6]

| Concern | Deep Agents: Run It Yourself | Managed Deep Agents: Managed Execution |
|---------|-----------------------------|---------------------------------------|
| Agent definition | `create_deep_agent` returns a compiled Agent | `define_deep_agent` returns a definition that the platform compiles |
| Business behavior | Configure the model, instructions, tools, and optional capabilities | Declare the model, instructions, tools, and optional capabilities in project files |
| Runtime infrastructure | Configure your own backend, store, checkpointer, and server | The managed runtime supplies these components and manages their lifecycle |
| Selection criterion | You need to run it yourself or customize underlying components | You want the ready-made Harness with less infrastructure work |

### Organizing Agent Behavior as Project Files

A research assistant can start with a small directory, adding only the capabilities its task needs:

```text
research-assistant/
├── agent.py          # Model and core configuration
├── instructions.md   # Behavioral instructions
├── tools/            # Business tools such as search
└── skills/           # Research workflows loaded on demand
```

`agent.py` declares the Agent definition, with tool implementations in `tools/`. Instructions describe how to complete research tasks, while Skills package reusable procedures. After the project is uploaded, the managed runtime assembles the definition into a runnable Agent.[6]

This follows the capability boundaries introduced earlier: **Tools provide actions, Skills describe procedures, the Harness manages execution, and the managed runtime supplies runtime infrastructure**. With these boundaries in mind, add capabilities as the task requires them; see the official documentation for complete configuration and deployment steps.[6]

> 💡 **Selection tip**: First decide whether you need Deep Agents' ready-made capabilities, then decide whether to use managed execution. Simple tool calling can continue to use a LangChain Agent; explicit control flow can use a custom LangGraph workflow. With MDA, developers still design business tool correctness, permission boundaries, and outcome evaluation. See [22.1 Agent Application Deployment Architecture](../chapter_22_deployment/01_deployment_architecture.md) for deployment choices.

> **Version note**: As of October 7, 2026, the official documentation labels MDA as Beta. This section focuses on responsibilities and project organization; consult the official documentation for current APIs and deployment requirements.[5][6]

---

## Deployment and MCP integration

LangSmith Agent Server provides durable execution, thread state, and streaming APIs. It can run on LangSmith Cloud, in your cloud, or on self-hosted infrastructure. Decide who owns the runtime, checkpoint storage, recovery, and identity checks. Consult the LangServe repository when maintaining an existing LangServe app, and the [Agent Server deployment guide](https://docs.langchain.com/langsmith/deploy-overview) and [Chapter 22](../chapter_22_deployment/README.md) when deploying a stateful agent.

For local development, register a Python graph export in `langgraph.json` and use `langgraph dev` to test threads, invocation, and streaming. Use `langgraph up` to check the PostgreSQL-backed environment before deployment. See [local development and testing](https://docs.langchain.com/langsmith/local-dev-testing).

LangChain's MCP integration discovers external tools and adapts them for `create_agent`. The official `MCPAdapter` interface requires `langchain[mcp]>=1.4.0` and is in beta. Check protocol compatibility between adapter and server, and keep connection-bound tools within the lifetime of the connection. See [LangChain MCP integration](https://docs.langchain.com/oss/python/langchain/mcp) and the [SDK example in 19.5](../chapter_19_protocol/05_practice_mcp_integration.md).

## Migrate an existing agent to v1

| Legacy pattern | v1 pattern |
|---|---|
| `AgentExecutor` + `create_openai_tools_agent` | `create_agent(model, tools, system_prompt=...)` |
| `langgraph.prebuilt.create_react_agent` | `langchain.agents.create_agent` |
| `input`, `chat_history`, `agent_scratchpad` | `messages` with an optional checkpointer and a `thread_id` per thread |
| Manually attached pre- or post-model hooks | Agent middleware |
| Legacy components such as `LLMChain` | LCEL for fixed pipelines, or `langchain-classic` for compatibility |

After migration, compare tool arguments, returned messages, and failure behavior. Start with [12.3](./03_langchain_agents.md) for a standard tool loop. Build a [LangGraph state graph](../chapter_langgraph/README.md) when you need direct control over state and execution order. See the [LangChain v1 migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1).

## References

[5] LangChain Team. [Deep Agents Overview](https://docs.langchain.com/oss/python/deepagents/overview). Accessed 2026-10-07.

[6] LangChain Team. [Managed Deep Agents Overview](https://docs.langchain.com/langsmith/python/managed-deep-agents-overview). Accessed 2026-10-07.
