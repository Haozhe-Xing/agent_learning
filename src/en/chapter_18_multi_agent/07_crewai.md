# 18.7 Framework Supplement: CrewAI (Role-Playing Multi-Agent Framework)

> 🎭 *CrewAI uses the simple abstraction of "role + task + crew" to make multi-Agent collaboration intuitive.*

---

CrewAI is a framework designed for multi-Agent collaboration, modeling collaboration through **role-playing**: each Agent has a clear **role / goal / backstory**, working like a well-divided team. It added the **Flows** event-driven workflow feature in 2025.

## Core Abstraction: Agent + Task + Crew

| Abstraction | Meaning | Your job |
|-------------|---------|----------|
| **Agent** | a role (researcher, editor) defined by role/goal/backstory | write the persona prompt, attach tools |
| **Task** | a concrete task, can declare dependencies | describe the task and expected output |
| **Crew** | a team, orchestrated by `Process` (sequential/hierarchical) | load agents + tasks, `kickoff` |

```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(role="Senior Researcher", goal="Collect accurate info",
                  backstory="10 years, data accuracy", verbose=True)
writer = Agent(role="Content Editor", goal="Turn research into readable article",
              backstory="writes for humans", verbose=True)

research_task = Task(description="Research {topic}", expected_output="Report", agent=researcher)
write_task = Task(description="Write article from research",
                  expected_output="Markdown article", agent=writer, context=[research_task])

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task],
            process=Process.sequential, verbose=True)
result = crew.kickoff(inputs={"topic": "LangGraph in production"})
```

## Two Execution Processes

| Process | Meaning | For |
|---------|---------|-----|
| `sequential` | tasks run in dependency order | clear pipeline |
| `hierarchical` | a manager Agent dispatches dynamically | flexible scheduling |

## Flows: precise orchestration

When you need **conditionals, loops, precise ordering**, use Flows (decorators `@start` / `@listen` / `@router`) — and you can embed a Crew inside a Flow for a hybrid.

## CrewAI vs LangGraph

| Dimension | CrewAI | LangGraph |
|-----------|--------|-----------|
| Barrier | low, declarative | higher (State/Node/Edge) |
| Scene | clear role division | precise control flow, loops, state |
| State mgmt | weak | strong (checkpoint + store) |
| HITL / debug | limited | native interrupt/resume |

**Choice**: clear roles → CrewAI; complex state/flow → LangGraph; fast prototype → CrewAI; production reliability → LangGraph.

---

*Back to chapter home: [Chapter 18: Multi-Agent Collaboration](./README.md)*
