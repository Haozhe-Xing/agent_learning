# 14.3 Architecture: Gateway / Agent Loop / Skills

> 🦞 *"Good architecture is the kind you can fork into six languages without breaking."*

---

## The Four-Layer Architecture

```
[Channel Adapters] ─► [Gateway] ─► [Agent Loop] ─► [Toolbox] ─► reply back
   (IncomingMessage)     (session)    (reasoning)     (execution)
```

### Layer 1: Channel Adapters

Each channel implements a uniform interface:

```ts
interface ChannelAdapter {
  start(): Promise<void>;
  stop(): Promise<void>;
  onMessage(handler: (msg: IncomingMessage) => void): void;
  send(to: string, reply: OutgoingMessage): Promise<void>;
}
```

Adding a channel = writing one new file; the core doesn't change.

### Layer 2: Gateway

Resolves sessions, pushes messages into history, triggers the Agent loop, and routes replies back through the original channel. It handles cross-channel identity, group-chat isolation, and interruption of long tasks.

### Layer 3: Agent Loop

```ts
async function agentLoop(input) {
  let context = assembleContext(input.session, input.userMessage);
  for (let step = 0; step < MAX_STEPS; step++) {
    const parsed = parseLlmOutput(await llm.stream(context));
    if (parsed.kind === 'final_answer') return parsed.text;
    if (parsed.kind === 'tool_call') {
      const decision = await permissions.check(parsed.toolCall);
      if (!decision.allowed) { context.appendError(...); continue; }
      await hooks.run('PreToolUse', parsed.toolCall);
      const result = await toolbox.run(parsed.toolCall);
      context.appendToolResult(parsed.toolCall.id, result);
    }
  }
}
```

Key points: streaming inference, context compression, permission + Hooks checks, error-tolerant retry (errors are written back so the LLM decides next step).

### Layer 4: Toolbox & Sandbox

Three sandbox tiers:

| Tier | Implementation | Use |
|------|----------------|-----|
| Default (no sandbox) | tools run in the main process | personal |
| Restricted shell | allow-list + arg validation | medium security |
| Docker | isolated container | production |

---

## Mapping to Chapter 8's Six Pillars

| Harness pillar | OpenClaw location |
|----------------|-------------------|
| Agent loop | `src/agent/loop.ts` |
| Tools | `src/agent/tools/` |
| Skills | Skills Registry + ClawHub |
| Memory | `src/memory/` (SQLite + summary) |
| Sandbox | default / restricted shell / Docker |
| Permissions | `permissions/` + Hooks + plan mode |

> The overlap is not coincidence — the "six pillars" are the minimal engineering form of an open-source harness like OpenClaw.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| Architecture | 4 layers: Channel Adapters → Gateway → Agent Loop → Toolbox |
| Protocol | all channels output `IncomingMessage`, input `OutgoingMessage` |
| Agent Loop | streaming + compression + permission + Hooks + error write-back |
| Sandbox | 3 tiers: none / restricted shell / Docker |
| Abstraction | stable enough to fork into 6 languages |

---

*Next section: [14.4 Multi-Channel Routing](./04_channels.md)*
