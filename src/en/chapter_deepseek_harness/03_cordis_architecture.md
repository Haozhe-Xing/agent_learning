# 15.3 Architecture: Cordis Microkernel & Plugin Topology

> 🐋 *"No privileged components — this is the fundamental difference from every other Agent framework."*

---

## What Cordis Is

Cordis (by the Koishi team) is a **plugin meta-framework** that does exactly one thing: let plugins be loaded, unloaded, depended on, and communicate with each other. It carries no business logic.

```typescript
import { Context } from 'cordis';

function greetPlugin(ctx: Context, config: { greeting: string }) {
  ctx.on('ready', () => console.log(config.greeting));
}
ctx.plugin(greetPlugin, { greeting: 'Hello, DeepSeek Harness' });
```

### Key abstractions

| Abstraction | Role |
|-------------|------|
| **Context (ctx)** | plugin runtime; `ctx.plugin` / `ctx.on` / `ctx.service` |
| **Service** | callable interface exposed to other plugins |
| **Event** | async message bus |
| **Effect** | one-time side effect on load |
| **Schema** | typed plugin config + defaults |

The dependency graph is **explicit** — a plugin declares what it needs before/after.

---

## DeepSeek Harness' Plugin Topology (standard profile)

![DeepSeek Harness plugin topology](../svg/chapter_deepseek_harness_03_cordis_topology.svg)

`core.agent.loop` being a plugin means you can swap it, run multiple loops, or remove it entirely — **something no other harness allows** (Claude Code / OpenClaw / Hermes all hard-code the loop in a privileged core).

---

## Communication: Services & Events

```typescript
// service
ctx.service('shell', {
  async run(command, opts = {}) { /* spawn child process */ }
});

// event
ctx.emit('tool.after', { name: 'shell', result });
ctx.on('tool.after', (e) => logger.log(...));
```

Stable context keys (`ctx.llm`, `ctx.tools`, `ctx.session`, `ctx.agentLoop`) are the contract third-party plugins rely on.

## Trajectory: git-style Replay & Fork

The event stream powers "trajectory replay": replay a whole session, fork at a step, audit what the Agent saw/did. A framework-level abstraction, more robust than hand-written logging.

## Paradigm Comparison

Chapters 12–15 offer architecture cases in different settings. Compare pluginization through common interface and lifecycle questions, rather than treating a diagram as evidence of an exclusive capability:

| Object | Interface and state checks | Failure checks on a fixed task |
|---|---|---|
| Event stream | What defines event types, ordering, and state updates? | How are replayed, duplicate, or missing events handled? |
| Execution isolation | Who checks permissions, environment, inputs, and outputs? | Do refusal, timeout, and cleanup preserve the contract? |
| State management | What is saved, and how are recovery formats versioned? | Can interruption and recovery duplicate side effects? |
| Tools | Do names, arguments, results, and errors match? | Are unauthorized tools and invalid arguments rejected? |
| Agent loop | Are entry, stopping, and resource ownership explicit? | Does replacement preserve success, failure, and exhaustion? |

Retain configuration, traces, and verdicts for the version actually tested. Mark unexecuted cases unknown. A plugin name does not establish replacement or migration acceptance.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| Cordis role | microkernel: load/unload/communicate only |
| Abstractions | Service / Event / Effect / Schema |
| Topology | LLM / Tool / Context / Skill / Sub-agent / Session / Sandbox / UI all plugins |
| Context keys | `ctx.llm` / `ctx.tools` / `ctx.session` stable |
| Trajectory | git-style replay / fork |
| Comparison method | Verify interfaces, state, permissions, cleanup, and fixed task results by version |

---

*Next section: [15.4 Plugin Development](./04_plugin_development.md)*
