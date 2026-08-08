# 14.5 How to Choose the Right Framework?

Framework selection is one of the key decisions for the success of an Agent project. With the release of the OpenAI Agents SDK in 2025 and the rapid iteration of various frameworks, the choices have become more abundant. This section provides a systematic decision-making framework.

![Agent Framework Selection Decision Guide](../svg/chapter_frameworks_05_choose.svg)

## Framework Capability Comparison Matrix

```python
framework_comparison = {
    "Framework": [
        "LangChain", "LangGraph", "CrewAI",
        "AutoGen", "OpenAI Agents SDK", "Dify", "Coze/n8n"
    ],
    
    "Learning Curve": ["Medium", "High", "Low", "Medium", "Low", "Low", "Very Low"],
    
    "Multi-Agent Support": [
        "Limited", "Native", "Core Feature",
        "Core Feature", "Native", "Supported", "Limited"
    ],
    
    "Workflow Complexity": [
        "Linear", "Complex Cycles", "Sequential/Flow",
        "Event-Driven", "Handoff", "Visual", "Visual"
    ],
    
    "MCP Support": [
        "Community Integration", "Community Integration", "Community Integration",
        "Community Integration", "Native Support", "Plugin", "Not Supported"
    ],
    
    "Production Ready": ["High", "High", "Medium", "Medium", "High", "Medium", "Low"],
    
    "Best Use Case": [
        "RAG/General Chains",
        "Stateful Workflows",
        "Role-Based Collaboration",
        "Code Generation/Dialogue",
        "Lightweight Production Agents",
        "Rapid Prototyping",
        "Non-Technical Users"
    ]
}

# Print comparison
for key, values in framework_comparison.items():
    print(f"\n{key}：")
    for framework, value in zip(framework_comparison["Framework"], values):
        if key != "Framework":
            print(f"  {framework}: {value}")
```

## Decision Tree

```python
def choose_framework(requirements: dict) -> str:
    """Select a framework based on requirements (2026 Updated Edition)"""
    
    # Non-technical team → Low-code
    if not requirements.get("technical_team"):
        return "Dify or Coze (Low-code Platform)"
    
    # Requires automatic code execution → AutoGen
    if requirements.get("code_execution"):
        return "AutoGen 0.4"
    
    # Lightweight Agent, fast deployment → OpenAI Agents SDK
    if (requirements.get("lightweight") and 
        not requirements.get("complex_control_flow")):
        return "OpenAI Agents SDK"
    
    # Multi-Agent with clear role division → CrewAI
    if (requirements.get("multi_agent") and 
        not requirements.get("complex_control_flow")):
        return "CrewAI"
    
    # Complex state management / cycles / Human-in-the-Loop → LangGraph
    if (requirements.get("complex_control_flow") or
        requirements.get("human_in_the_loop") or
        requirements.get("stateful_workflow")):
        return "LangGraph"
    
    # Standard RAG / Single Agent → LangChain
    return "LangChain"


# Test decisions
scenarios = [
    {
        "name": "Enterprise Knowledge Base Q&A",
        "technical_team": True,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "Automated Software Development Assistant",
        "technical_team": True,
        "multi_agent": True,
        "code_execution": True,
        "complex_control_flow": True,
        "lightweight": False
    },
    {
        "name": "Content Creation Team",
        "technical_team": True,
        "multi_agent": True,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "Customer Service Automation (Business Configuration)",
        "technical_team": False,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": False
    },
    {
        "name": "Quickly Build Tool-Calling Agent",
        "technical_team": True,
        "multi_agent": False,
        "code_execution": False,
        "complex_control_flow": False,
        "lightweight": True
    }
]

print("Framework Selection Recommendations：\n")
for scenario in scenarios:
    name = scenario["name"]  # Use [] access instead of pop() to avoid modifying the original dict
    # Build a requirements dict without 'name' and pass it to the decision function
    requirements = {k: v for k, v in scenario.items() if k != "name"}
    framework = choose_framework(requirements)
    print(f"Scenario：{name}")
    print(f"Recommendation：{framework}\n")
```

## Strategy for Real Projects: Composition, Not Binding

> ⚠️ **Honest Note**: In the past, this section used a `HybridAgentSystem` class to "demonstrate" multi-framework composition, but its `build()` method body was `pass` — no real code at all, a filler approach of "wrapping an empty shell as a production-ready system." This section now provides an **actionable composition decision matrix** and explicitly identifies the seam points you must implement yourself for each combination.

Real production systems do often mix multiple frameworks, but the combination depends on "which part is the bottleneck," not on importing every framework. Below are composition suggestions grouped by bottleneck:

| Your Bottleneck | Primary Framework | Auxiliary Framework | Real Seam Points (Must Implement Yourself) |
|-----------------|-------------------|---------------------|--------------------------------------------|
| Complex State / Cycles / Human-in-the-Loop | LangGraph | MCP (Tool Standardization) | Model business state as `TypedDict`, use conditional edges for approval/rollback |
| Multi-Role Pipeline (Researcher→Editor→Reviewer) | CrewAI / Custom | — | Role prompts + task dependencies; note that CrewAI does not include built-in code execution |
| Agent Needs to Actually Run Code | AutoGen | Docker Sandbox | Sandbox executor + execute-error-correct loop |
| Lightweight Tool-Calling Agent, Fast Deployment | OpenAI Agents SDK | MCP Native | Handoff definitions, tool schemas |
| RAG / Knowledge Base Q&A | LangChain | Vector DB (Chroma, etc.) | Retriever + reranking + citation tracing |
| Non-Technical Team Rapid Validation | Dify / Coze (Low-code) | — | Visual workflow orchestration, channel integration |

Two hard principles for composition:

1. **Decouple business logic from frameworks**: Tool functions, domain models, and validation logic should be written as framework-agnostic pure functions; the framework is only responsible for orchestration. This way, if you switch frameworks later (e.g., migrating from CrewAI to LangGraph), you don't need to rewrite business code.
2. **Use MCP to standardize tool interfaces**: Regardless of which framework you use, tools should follow the MCP standard to reduce switching and integration costs.

> For a minimal example of "how multiple frameworks collaborate in one real system," see Chapter 16 `examples/dev_team/` which uses LangGraph for orchestration, and the unified base `reference-agent/` which uses a Provider abstraction to isolate models and tools — neither engages in framework stacking just for the sake of it.

## Final Recommendations

Core principles for framework selection:

1. **Start simple**: First try the OpenAI Agents SDK or LangChain + direct API calls; don't introduce complex frameworks if a simple approach suffices
2. **Upgrade based on bottlenecks**: When you find you need state management → introduce LangGraph; need multi-role collaboration → consider CrewAI; need code execution → consider AutoGen
3. **Embrace standard protocols**: Use MCP to standardize tool interfaces and reduce framework switching costs; keep an eye on the A2A protocol for Agent-to-Agent interoperability
4. **Maintain framework-agnostic code**: Business logic should not be tightly coupled to any framework; tool functions should remain generic
5. **Prioritize debugging and observability**: In production, prefer solutions with good logging and observability (LangSmith, Dify, etc. all provide strong observability capabilities)
6. **Community and ecosystem**: Choose actively maintained frameworks (check GitHub activity); the most active in 2025: LangGraph, CrewAI, OpenAI Agents SDK

---

## Summary

Overview of major frameworks:

| Framework | Core Strength | Recommended Use Case |
|-----------|--------------|---------------------|
| LangChain | Rich ecosystem, strong RAG | General Agents, rapid development |
| LangGraph | State management, complex workflows | Production-grade stateful Agents |
| CrewAI | Simple multi-agent + Flows | Tasks with clear role divisions |
| AutoGen 0.4 | Event-driven, code execution | Programming automation tasks |
| OpenAI Agents SDK | Lightweight, MCP native | Rapidly build production Agents |
| Dify/Coze | Low-code visual tools | Non-technical team rapid validation |

---

## 📝 Chapter Exercises

After reading this chapter, close the book and answer the following questions in your own words first, then expand the reference answers for comparison.

**Exercise 1 (Concept)**: AutoGPT and BabyAGI ignited the "autonomous Agent" craze in 2023, but this chapter says they have "limited practicality in production environments." Name at least three core problems they exposed as summarized in this chapter, and explain how modern frameworks (e.g., designs with `max_iterations` and Human-in-the-Loop) respond to these lessons.

<details>
<summary>Reference Answer</summary>

The value of AutoGPT/BabyAGI lies in "proof of concept" — they demonstrated that LLMs can autonomously decompose and execute complex tasks, but they exposed the typical problems of fully autonomous Agents:

1. **Goal Drift**: During execution, the Agent gradually deviates from the original goal. For example, if the goal is "write a blog post," the Agent may go off searching for materials, researching writing techniques, investigating tools… and forget to write the article.
   → Modern response: Define goals **clearly and with boundaries** ("list the top 5 most common complaints" rather than "make the product better"), and verify each step against the original goal.

2. **Infinite Loops / No Termination Condition**: "Keep running until done" may never stop in practice.
   → Modern response: Enforce `max_iterations` / `max_steps` / `max_turns` and budget caps — AutoGen and CrewAI, as seen in this chapter, all have such limits built in.

3. **Limited Task Decomposition Ability**: The quality of automatic planning by the model is far inferior to a well-designed human workflow.
   → Modern response: Human-assisted planning + Agent execution, or use LangGraph to explicitly draw the flow as a graph, or use CrewAI to explicitly define roles and tasks, rather than letting the model improvise freely.

4. **Error Propagation**: Small errors in early steps get amplified in subsequent steps.
   → Modern response: Step-by-step verification + rollback mechanisms, as well as **Human-in-the-Loop** — letting humans confirm at key nodes to prevent errors from snowballing. Fully autonomous Agents carry high risk in production; "human supervision + Agent execution" is the currently viable approach.

In one sentence: What AutoGPT taught the industry is not "how to build fully autonomous Agents," but "why fully autonomous Agents are unreliable," thus driving the design of modern frameworks that are **bounded, controllable, and observable**.

</details>

**Exercise 2 (Differentiation)**: CrewAI and AutoGen are both multi-Agent frameworks, but their positioning is very different. Someone says: "Multi-Agent frameworks are all pretty similar, just pick one." Refute this using the content of this chapter: compare CrewAI and AutoGen from three angles — "core philosophy," "whether they support code execution," and "suitable scenarios" — and give one task most suitable for each.

<details>
<summary>Reference Answer</summary>

"Just pick one" is wrong — the design philosophies and killer features of the two are completely different; picking the wrong one will make your project twice as hard for half the result.

| Comparison Angle | CrewAI | AutoGen |
|-----------------|--------|---------|
| **Core Philosophy** | Role-playing + task flow: First define Agents (role/goal/backstory) and Tasks, execute in a predetermined flow (sequential or hierarchical) | Free dialogue among Agents: Treat each Agent as a "meeting participant," advancing through natural language back-and-forth discussion |
| **Code Execution** | ❌ Not built-in | ✅ Built-in sandbox executor (Docker / local), capable of "generate code → execute → review errors → auto-correct" — this is its killer feature |
| **Suitable Scenarios** | Pipeline tasks with clearly defined roles and a pre-orchestrated flow | Programming/data analysis tasks that require actually running code, or scenarios needing flexible multi-round discussion |

**One task most suitable for each:**
- **CrewAI is best for**: Content creation pipeline — "Researcher" gathers materials → "Editor" writes the article → "Reviewer" scores and revises. Roles are clear, the flow is fixed; CrewAI's declarative definition is the simplest approach.
- **AutoGen is best for**: Automated programming tasks — "Programmer" writes code, "Code Executor" runs it in a Docker sandbox, and upon error, the programmer reads the error log and corrects it automatically. This "generate-execute-correct" closed loop is unique to AutoGen; CrewAI cannot do it.

**Key decision point**: "Does the Agent need to actually run code?" — If yes, AutoGen is almost the only choice; if it's just role-based pipeline execution, CrewAI is simpler and more intuitive. So frameworks are absolutely not something you "just pick at random."

</details>

**Exercise 3 (Hands-On)**: Your team receives this requirement: "Build an internal IT ticket assistant for a company. Employees ask questions via WeChat, the assistant queries a knowledge base to answer common questions, and complex issues are escalated to human support. The team consists only of a product manager and an operations person with no dedicated developers, and the deadline is one week to launch a trial." Based on this chapter's "Framework Selection Decision Tree" and selection principles, provide your framework choice with a complete explanation of the rationale; then supplement: if in six months this assistant needs to support "multi-step approval workflows + deep integration with the internal ticketing system," how would you evolve the technical solution?

<details>
<summary>Reference Answer</summary>

**Phase 1 Choice: Low-code Platform (Coze or Dify).**

Following this chapter's decision tree, the first decision node is "Is there a technical team?" — this requirement explicitly states "no dedicated developers," so the decision tree directly outputs "Dify or Coze (Low-code Platform)." Let's verify against each specific constraint:

| Requirement | Why Low-Code Fits |
|-------------|-------------------|
| No developers; product/operations-led | Low-code uses graphical drag-and-drop; non-technical users can build it |
| One-week launch | Low-code prides itself on "1-3 days to prototype" — easily meets the deadline |
| WeChat channel | **Coze** natively supports publishing to WeChat, Feishu, and other IM platforms — more convenient than Dify here |
| Query knowledge base for common questions | Both have built-in knowledge base / RAG retrieval |
| Escalate complex issues to human support | Can implement "detect complex issue → escalate to human" using workflow conditional branches |

**Conclusion**: Prefer **Coze** (because of the need to distribute to WeChat, which is its strength); if the enterprise requires on-premises deployment and data must stay within the intranet, choose **Dify** (open-source, self-hostable).

**Phase 2 Evolution (Six months later: multi-step approval + deep ticketing system integration):**

Evolve following this chapter's principles of "upgrade based on bottlenecks" and "hybrid solutions," rather than jumping to a heavyweight framework from the start:

1. **Identify the bottleneck**: The low-code platform's complex state management, multi-step approval workflows (which may involve loops, rollbacks, counter-signatures), and deep integration with the internal ticketing system will exceed the comfort zone of visual orchestration.
2. **Migrate core logic to LangGraph**: Multi-step approval is essentially a "stateful workflow with conditional branches and cycles" — this is exactly LangGraph's strength (strong state management + cycle control + Human-in-the-Loop for approval confirmation).
3. **Keep low-code for non-core parts**: Following the "hybrid solution" approach, continue using Coze/Dify for the front-end dialogue and simple Q&A, while rewriting the approval and other core processes in LangGraph, with both integrating via APIs.
4. **Use MCP to standardize tool interfaces**: Build the ticketing system integration as standardized tools (following MCP), reducing the cost of future framework changes or new system additions — echoing this chapter's principle of "embrace standard protocols, maintain framework-agnostic code."

One-sentence summary of the evolution approach: **First use low-code to quickly validate the product direction; after successful validation, migrate core processes to a code framework (LangGraph) based on real bottlenecks, while keeping business logic decoupled from the framework.**

</details>

---

*Next Chapter: [Chapter 16 Multi-Agent Collaboration](../chapter_multi_agent/README.md)*
