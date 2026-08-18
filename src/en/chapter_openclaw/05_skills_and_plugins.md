# 14.5 Skills & Plugin Ecosystem: ClawHub and Community Contributions

> 🦞 *"SKILL.md is not a document — it's a public contract."*

---

## SKILL.md: OpenClaw's Extension Contract

Each Skill is a directory with `SKILL.md` (plus optional scripts):

```
~/.openclaw/skills/
├── send-email/
│   ├── SKILL.md
│   └── send.ts
└── flight-search/
    ├── SKILL.md
    └── search.py
```

### Structure

```markdown
---
name: send-email
description: Send an email via Gmail. Use when the user asks to "send/reply email".
tools: [gmail.send]
permissions: [gmail.send]
version: 0.1.0
author: yourname
---

# Trigger
User asks to send / reply / broadcast email.

# Flow
1. Collect recipient, subject, body
2. Call gmail.send
3. Return result (message id / success)

# Constraints
- Plain text by default unless HTML requested
- Confirm recipient & subject before sending
```

### The 4 key fields

| Field | Purpose | Who reads it |
|-------|---------|-------------|
| `name` | unique id | registry |
| `description` | when to use it | the LLM |
| `tools` | required tools | permission system |
| `permissions` | required permissions | permission system |

> 📌 `description` is a **retrieval index for the LLM** — write it well so the Agent triggers the Skill at the right moment.

---

## ClawHub: The Skill Marketplace

```bash
openclaw skills search "email"       # browse
openclaw skills install send-email   # install
openclaw skills publish ./my-skill   # publish (community)
```

### Three ecosystem layers

| Layer | Content | Contributor |
|-------|---------|-------------|
| Core Skills | built-in (email/calendar/flight/notes) | OpenClaw team |
| Community Skills | listed on ClawHub | community devs |
| Private Skills | `~/.openclaw/skills/` | yourself |

---

## Community Forks: Proof of Sound Abstraction

| Project | Language | Note |
|---------|----------|------|
| OpenClaw (main) | TypeScript | official |
| nanobot | Python (~4000 lines) | minimal, teaching |
| ZeroClaw | Rust | system-level perf |
| NanoClaw | Go + Apple containers | macOS isolation |
| IronClaw | Rust + WASM | WASM sandbox |
| NullClaw | Zig | 678KB static binary |

> ⚠️ Sizes per each repo's `main` branch README.

The core abstraction is small and stable enough that re-implementing it in another language is "translation," not "redesign."

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| SKILL.md | frontmatter + trigger + flow + constraints |
| 4 fields | name / description / tools / permissions |
| ClawHub | search / install / publish |
| 3 layers | core / community / private |
| Forks | 6 languages = sound abstraction |

---

*Next section: [14.6 Practice: Build a Personal Assistant](./06_practice.md)*
