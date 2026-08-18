# 16.1 The Prehistory of Industrial Harnesses: From AutoGPT to Claude Code

> 🛠️ *"Claude Code didn't appear out of thin air — it stands on the shoulders of AutoGPT, BabyAGI, and OpenHands."*

---

## Why This Chapter Starts with AutoGPT

Many readers first encounter Claude Code and feel it "suddenly appeared." It didn't — it is the latest link in a chain of Agent frameworks and harnesses. Understanding that chain is how you grasp both what Claude Code *did that was new* and *which old paradigms it inherited*.

This section traces that lineage — from AutoGPT and BabyAGI in 2023 to Claude Code in 2026 — so you understand:

- Claude Code is **not** the first "AI coding tool" — what it pioneered was the "industrial Claude-family IDE paradigm";
- Each project along the way contributed one or two core paradigms (event stream / state graph / code-as-action …);
- Those paradigms still live inside Claude Code today.

---

## AutoGPT (April 2023) — "Agent" Goes Mainstream

**AutoGPT** (by Significant Gravitas) was essentially "GPT-4 + a toolbox + a persistent loop." Its significance was not what it *accomplished* but that it put the word **"Agent"** into the mainstream.

```python
# AutoGPT's early main loop (simplified reconstruction)
while True:
    response = llm(prompt=PROMPT_TEMPLATE.format(history=...))
    action = parse(response)
    if action.type == "finish":
        return action.result
    result = execute(action)
    history.append((action, result))
```

### The paradigm legacy

Although AutoGPT was quickly shown unable to complete serious tasks (loops drift, token cost explodes, no persistent state), it left several abstractions that still hold:

- **Agent Loop** — the "think-act-observe" cycle as the minimal executable unit;
- **Tool Taxonomy** — a unified tool-description schema;
- **Memory Layer** — conversation history abstracted as vector/summary store;
- **Goal-Directed Loop** — a goal plus LLM-generated subgoals.

> 📌 **AutoGPT changed terminology, not engineering.** Before it: "AI assistant" / "chatbot." After it: "Agent" became a product category.

---

## BabyAGI (April 2023) — The Seed of Task Management

**BabyAGI** (by Yohei Nakajima) took a different approach days later:

```
task queue → take next → LLM executes (with context) → store result → regenerate tasks → loop
```

Its core innovation was **task decomposition + task queue** — turning AutoGPT's fuzzy "goal" into a concrete *task list*.

Two abstractions it left behind:

1. **Task as a first-class citizen** — a data structure (description / status / result), not a string in a prompt;
2. **Re-prioritization loop** — the LLM re-ranks task priority each cycle.

Both later reappear in LangChain's PlanAndExecute, AutoGen's Task abstraction, and Claude Code's sub-agents.

---

## OpenHands (formerly OpenDevin, March 2024) — The First "Action-Oriented" Open Agent

**OpenHands** (All-Hands-AI) was the first mature open-source "action-oriented" Agent for software engineering — read code, edit code, run commands in a real environment.

It contributed at least five abstractions that every later action-Agent inherited:

1. **EventStream** — a central bus where every action/observation is a typed event; state = event log;
2. **Sandbox Runtime** — Docker sandbox by default (`cap-drop ALL`, `no-new-privileges`);
3. **LiteLLM model abstraction** — 100+ models behind one interface;
4. **LLMSummarizingCondenser** — context compression after N turns;
5. **Action Type System** — each action (`FileWriteAction`, `CmdRunAction`, …) is a Pydantic model.

### Direct descendants

| Project | What it inherited |
|---------|------------------|
| Claude Code | Event stream + sandbox + action types |
| Aider | Terminal-native + LiteLLM |
| Continue.dev | Multi-model + action types |

---

## AutoGen (late 2023 – 2024) — Multi-Agent Dialogue

**AutoGen** (Microsoft Research) framed multi-Agent collaboration as *conversation* — each Agent is an actor, and actors collaborate through messages.

```python
from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent("assistant", llm_config=...)
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"use_docker": True})
user_proxy.initiate_chat(assistant, message="Find perf issues in ./src/")
```

Claude Code didn't adopt AutoGen's code, but inherited its ideas: role-based agents, group-chat patterns, and code execution in a sandbox.

---

## CrewAI (early 2024) — Role-Playing

**CrewAI** models collaboration as a *team*: `Agent` (role) + `Task` + `Crew`/`Flow`. It made "role-playing" a productized path — but Claude Code uses **engineering roles** (Sub-Agent, Bash Runner) rather than *human* roles, which is more predictable and debuggable.

---

## The Whole Lineage in One Picture

```
2023 Q1 ── 2024 Q1 ── 2024 Q3 ── 2025 ── 2026
AutoGPT   OpenHands   MCP       OpenClaw   Claude Code
BabyAGI   Aider       protocol  Hermes     DeepSeek Harness
   │          │          │         │            │
   └────── 留下了 "Agent Loop + 工具 + 状态 + 沙箱" 四大支柱 ──────┘
```

**Key takeaways:**

1. No modern harness is invented from scratch — they all stand on AutoGPT / BabyAGI / OpenHands.
2. Post-2024 differentiation is **product form** — same core, different surface (IDE / chat app / self-evolving / pluginized).
3. The key to understanding Claude Code is looking back at OpenHands — without its event stream + action types, Claude Code's six-layer architecture wouldn't look the way it does.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| AutoGPT | Mainstreamed "Agent"; left Agent Loop / Tool Taxonomy / Memory / Goal Loop |
| BabyAGI | Task as first-class data structure + re-prioritization |
| OpenHands | First action-oriented open Agent; event stream + sandbox + LiteLLM + action types |
| AutoGen | Conversation-as-orchestration for multi-Agent |
| CrewAI | Role-playing; Claude Code chose engineering roles instead |
| Key insight | Claude Code is not invented from scratch — it stands on 5 years of open-source accumulation |

---

*Next section: [16.2 Getting to Know Claude Code: From Zero to Hands-On](./02_introduction.md)*
