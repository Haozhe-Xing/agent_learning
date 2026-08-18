# 19.6 Framework Supplement: AutoGen (Multi-Agent Dialogue Framework)

> 💬 *"Letting Agents discuss in natural language is closer to how humans work than routing them through fixed nodes."*

---

AutoGen is Microsoft's multi-Agent framework; its core innovation is advancing tasks **through Agent-to-Agent "conversation"** rather than a call chain. Each Agent is a "meeting participant" discussing in natural language.

> ⚠️ Version note: In late 2024 the team split into Microsoft's **AutoGen 0.4** (full rewrite, event-driven) and community-maintained **AG2** (0.2 API). This section targets 0.4.

AutoGen 0.4's design threads: async messaging + event-driven; pluggable runtime; Pydantic-typed messages; and its killer feature — **code auto-execution** (generate code → run in sandbox → feed result back → fix).

## Killer Feature: Code Execution Sandbox

| Executor | Isolation | Use |
|----------|-----------|-----|
| `DockerCommandLineCodeExecutor` | Docker sandbox | production |
| `LocalCommandLineCodeExecutor` | host | dev only (no isolation) |

```python
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent, AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

code_executor = DockerCommandLineCodeExecutor(image="python:3.12-slim", timeout=60)
executor_agent = CodeExecutorAgent("executor", code_executor=code_executor)
coder = AssistantAgent("coder", system_message="Write Python, fix on error, say TERMINATE when done",
                       model_client=model_client)
async with code_executor:
    team = RoundRobinGroupChat([coder, executor_agent],
                               termination_condition=TextMentionTermination("TERMINATE"), max_turns=10)
    await team.run(task="Download and analyze the iris dataset")
```

## Group Chat Modes

| Mode | Scheduling | For |
|------|-----------|-----|
| `RoundRobinGroupChat` | take turns | fixed role order |
| `SelectorGroupChat` | LLM picks next speaker | roles shift by stage |

## AutoGen vs CrewAI

| Dimension | AutoGen | CrewAI |
|-----------|---------|--------|
| Core idea | free dialogue | role + task flow |
| Code execution | ✅ built-in sandbox | ❌ |
| Flexibility | high | medium |
| Cost | higher (multi-turn) | lower |
| Scene | code gen/debug, data analysis | content creation, pipeline |

**Choice**: generate-and-run code → AutoGen; role-defined pipeline → CrewAI; flexible discussion → AutoGen.

---

*Back to chapter home: [Chapter 19: Agent Communication Protocols](./README.md)*
