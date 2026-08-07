# Chapter 15: Deep Dive into Claude Code: From Usage to Source Code

> 🛠️ *"Tools are not just for using — they are for understanding. Truly mastering a tool means you can predict its limits, control its behavior, and even reshape its direction."*  
> — Adapted from Richard Feynman

---

## Chapter Introduction

On March 31, 2026, an accident shook the AI tools community: Anthropic's terminal AI programming tool, Claude Code, accidentally exposed its complete, unminified TypeScript source code stored in an R2 bucket due to a misconfigured npm package source map. Community engineers completed a preliminary analysis of nearly 100,000 lines of code within hours, revealing a system far more complex than it appeared on the surface — a six-layer hierarchical architecture, a meticulously designed 915-line System Prompt, a 6-stage permission decision pipeline, and previously undisclosed operating modes such as ULTRAPLAN and KAIROS.

This is one of the rare "outside-in" chapters in this book. In previous chapters, we learned how to **build** Agent systems using frameworks like LangChain and LangGraph. In this chapter, we study how Anthropic's engineers designed an Agent tool **for AI itself** that can work reliably in real software projects. Claude Code is both a powerful development tool you can use directly and a first-hand reference for top-tier Agent system design — its careful use of Prompt Cache, its layered governance of permission boundaries, and its three-level context compression strategy are all worth studying closely.

Through this chapter, you will gain value on two levels: **practical level** — you will truly master Claude Code, from basic installation to MCP integration, Hooks automation, and fine-grained CLAUDE.md configuration, significantly boosting your daily development efficiency; **architectural level** — you will see how a production-grade AI programming Agent handles permissions, manages context, and orchestrates multi-Agent collaboration, and these design ideas can be directly applied to your own Agent systems.

---

## Chapter Content Overview

| Section | Content | What You'll Learn |
|---------|---------|-------------------|
| 15.1 Getting to Know Claude Code: From Zero to Hands-On | Installation, core interaction modes, common commands and shortcuts, fundamental differences from Copilot/Cursor | Quickly get started with Claude Code and understand how it differs fundamentally from traditional code completion tools as an Agent tool |
| 15.2 Deep Dive into Core Architecture | Six-layer hierarchical architecture, QueryEngine main loop, React+Ink terminal UI, three-level context compression mechanism | Understand Claude Code's overall design philosophy and how Agent systems maintain stability at the engineering level |
| 15.3 Source Code Decoded: System Prompt and Permission Engineering | Four module types in the 915-line System Prompt, static/dynamic zone separation, Prompt Cache reducing costs by 90%, 6-stage permission decision pipeline | Master industrial-grade System Prompt design patterns and understand the correct way to model permission systems |
| 15.4 Advanced Usage: MCP, Hooks, and Skills | MCP server integration with the external world, Hooks event-driven automation (PreToolUse/PostToolUse), Skills reusable capability packages, sub-Agent orchestration | Use extension mechanisms to turn Claude Code into a team-specific workflow engine |
| 15.5 Production Practice: Using Claude Code Effectively in Teams | CLAUDE.md best practices, team configuration sharing, cost control, security considerations and vulnerability review | Deploy and use Claude Code stably, efficiently, and securely in real team environments |

---

## Reading Recommendations

This chapter is suitable for the following readers:

- ✅ **Engineers looking to boost productivity**: want to truly use AI tools rather than just occasionally asking questions — read from 15.1 in order
- ✅ **Agent system builders**: building your own Agent systems and want to study the engineering implementation details of top teams as a reference — focus on 15.2 and 15.3
- ✅ **Team leads/architects**: need to evaluate and promote AI programming tools within a team — focus on 15.4 and 15.5

**Prerequisite knowledge**: It is recommended to read Chapter 7 (Context Engineering) and Chapter 8 (Harness Engineering) first to have a basic understanding of Agent engineering control before reading the architectural analysis sections of this chapter. Sections 15.1 and 15.4 are relatively independent; readers not interested in source code analysis can start directly with the practical sections.

> 💡 **Special note**: Some content in this chapter is based on community analysis following the accidental source code exposure incident in March 2026. Anthropic fixed the disclosed security vulnerabilities in Claude Code v2.1.90 (April 4, 2026). The security analysis in this chapter is for educational reference only; please do not use it for malicious purposes.

---

*Next section: [15.1 Getting to Know Claude Code: From Zero to Hands-On](./01_introduction.md)*
