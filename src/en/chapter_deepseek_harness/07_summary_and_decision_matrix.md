# 17.7 Summary: The 6-Harness Decision Matrix

> 🐋 *"One table to choose your entire Agent tool stack."*

---

## Part III Recap

```
Part III: Framework Practice (6 chapters)
│
├─ Code frameworks (to BUILD an Agent)
│   ├─ Ch.12 LangChain — assembly library (chains / tools / Agent / LCEL)
│   └─ Ch.13 LangGraph — stateful graph (checkpoints / routing / HITL)
│
└─ Action harnesses (to USE an Agent)
    ├─ Ch.14 OpenClaw       — cross-platform personal assistant (multi-channel)
    ├─ Ch.15 Hermes Agent   — self-evolving assistant (self-evolution)
    ├─ Ch.16 Claude Code    — industrial IDE (coding)
    └─ Ch.17 DeepSeek Harness — everything-is-a-plugin foundation (platform)
```

---

## The Decision Matrix

| Your need | Best fit | Second |
|-----------|----------|--------|
| Fast multi-model/multi-tool integration | LangChain | OpenClaw |
| Complex stateful + human-in-the-loop | LangGraph | LangChain + LCEL |
| Agent in WhatsApp/Telegram | OpenClaw | Hermes |
| Agent that auto-learns new skills | Hermes | (no second yet) |
| Agent in a coding IDE | Claude Code | DeepSeek Harness |
| Long-term controllable Agent platform | DeepSeek Harness | OpenClaw + DIY |
| Teaching / learning principles | Claude Code (source) + LangChain | OpenClaw |
| Avoid commercial products entirely | DeepSeek + OpenClaw + Hermes | DIY |
| Production SaaS Agent | LangChain + LangGraph + LangSmith | OpenClaw (+ops) |
| Research / benchmarking | LangChain + minimal | DeepSeek `--profile minimal` |

## Decision Tree

```
Q1: "Use" an Agent, or "build" one?
├─ use → harness subtree
│   ├─ chat apps? → OpenClaw (Ch.14)
│   ├─ grow smarter? → Hermes (Ch.15)
│   ├─ software-engineering IDE? → Claude Code (Ch.16)
│   └─ long-term controllable platform? → DeepSeek Harness (Ch.17)
└─ build → code-framework subtree
    ├─ complex flows + state? → LangGraph (Ch.13)
    ├─ fast multi-tool? → LangChain (Ch.12)
    └─ DIY? → read any "borrowing" section + Ch.8
```

## 6-Framework Comparison

| Dimension | LangChain | LangGraph | OpenClaw | Hermes | Claude Code | DeepSeek Harness |
|-----------|-----------|-----------|----------|--------|-------------|------------------|
| Type | framework | framework | harness | harness | harness | harness |
| License | MIT | MIT | MIT | MIT | source-avail | MIT |
| Core paradigm | Runnable chain | state graph | multi-channel | self-evolving | industrial IDE | all-plugins |
| Swappable kernel | — | — | ❌ | ❌ | ❌ | ✅ |
| Model-agnostic | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Customization | high (code) | high (code) | mid (skill) | mid (skill) | low (config) | very high (plugin) |
| Skill protocol | tool ABC | tool ABC | SKILL.md | SKILL.md | SKILL.md | SKILL.md |

## A Typical 12-Month Evolution

```
Month 1      → LangChain: build from scratch (learn principles)
Month 2–3    → LangGraph: state, loops, HITL (learn engineering)
Month 4–6    → Claude Code: daily IDE companion
Month 6–12   → OpenClaw + Hermes: personal assistant takes over chores
            → DeepSeek Harness: long-term controllable open foundation
```

## Three Typical Stacks

| Scenario | Stack |
|----------|-------|
| Research / teaching | LangChain + LangGraph + local LLM + Claude Code source as exemplar |
| Consumer / personal | OpenClaw + Hermes + any LLM + Docker |
| Engineering team | Claude Code + LangChain/LangGraph + MCP + LangSmith |
| Long-term autonomy | DeepSeek Harness + self-written plugins + fallback chain |

---

## Part III Conclusion

Six chapters laid out the entire "action-Agent" landscape: cognition (framework vs harness), library (LangChain/LangGraph), product (OpenClaw/Hermes/Claude Code/DeepSeek Harness), paradigm (they're the same event-stream + sandbox + state + tools + permissions in different surfaces), and engineering (read any one, see all four).

Next: **Part IV — Multi-Agent Systems (Chapter 18+)**.

---

*Back to chapter home: [Chapter 17: DeepSeek Harness](./README.md)*
