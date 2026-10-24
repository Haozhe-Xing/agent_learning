# Chapter 14: OpenClaw — Cross-Platform Personal AI Assistant

> 🦞 *"Your lobster butler, already living in every chat app you use."*

---

## Chapter Introduction

Most action-oriented Agents (like Claude Code or Hermes) live in a CLI/TUI. If you want an Agent that lives **inside the chat apps you already use** — WhatsApp, Telegram, Discord, Slack, Signal — **OpenClaw** is the most complete implementation in the current ecosystem.

OpenClaw is an **open-source personal AI assistant** started by Peter Steinberger in December 2025 as a weekend project. It was first named `Clawdbot`, then `Moltbot`, and finally `OpenClaw` in January 2026 (to avoid trademark issues and emphasize open-source). Its growth has been phenomenal — it became one of the fastest-growing open-source projects on GitHub. OpenClaw's rapid spread turned the "personal Agent" from a geek toy into an everyday tool.

This chapter does three things:

1. **Dissect OpenClaw's engineering skeleton** — what modules make up "an Agent in a chat app" and how messages flow through the gateway and Agent loop;
2. **Demonstrate 4 deployment methods** — `npx`, install script, source build, Docker sandbox;
3. **Map its subsystems to Chapter 8's "six engineering pillars"** — so you see how a real production project instantiates the abstractions you learned earlier.

---

## Chapter Content Overview

| Section | Content | What You'll Learn |
|---------|---------|-------------------|
| 14.1 OpenClaw Panorama | Origin, naming history, positioning | OpenClaw's place in the harness spectrum |
| 14.2 Installation & 4 Deployment Modes | npx / install script / source / Docker | Get it running on macOS/Linux/Windows |
| 14.3 Architecture: Gateway / Agent Loop / Skills | Four-layer architecture, message flow, plugins | Read OpenClaw's engineering skeleton |
| 14.4 Multi-Channel Routing | 5 platforms, group/DM policy, cross-channel identity | Take over all chat entrances |
| 14.5 Skills & Plugin Ecosystem | SKILL.md contract, ClawHub, community forks | Find, customize, publish OpenClaw skills |
| 14.6 Practice: Build a Personal Assistant | End-to-end 5-step build | Ship a 24/7 personal assistant |
| 14.7 Lessons for Engineers | 5 engineering insights | Bring OpenClaw's philosophy to your system |

---

## Reading Recommendations

- ✅ **Want to actually "use" an Agent** — read from 14.1
- ✅ **Cross-platform developers** — focus on 14.3, 14.4
- ✅ **Engineers studying open-source harnesses** — focus on 14.7

**Prerequisites**: Chapter 8 (Harness Engineering) and Chapter 9 (Skill System) — their abstractions land concretely in OpenClaw's subsystems.

> 📌 **Chapter relationships**: OpenClaw is the **consumer-product route** of the action-Agent spectrum; Claude Code (Ch.16) is the **industrial IDE route**; DeepSeek Harness (Ch.17) is the **developer-workshop route**. After this chapter, read Ch.15 Hermes — same "multi-channel" lineage, but more aggressive on self-evolution.

---

## Citation Convention

Verifiable facts (commits, versions, install commands, licenses, file paths) are anchored to the OpenClaw `main` branch and `openclaw.ai`. Time-varying numbers (stars, extension counts) are **not cited**; versions are anchored to "as of August 2026" where relevant.

> 💡 **Core insight**: OpenClaw changed not "Agent capability" but "how an Agent reaches the user" — when the user key becomes `@bot` in a chat box, the Agent enters daily life.

---

*Next section: [14.1 OpenClaw Panorama: From Clawdbot to OpenClaw](./01_history_and_positioning.md)*
