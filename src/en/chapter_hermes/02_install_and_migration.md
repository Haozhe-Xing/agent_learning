# 15.2 Installation & Migration (from OpenClaw)

> ☤ *"From 'run it' to 'migrate from OpenClaw' — both in one command."*

---

## 5 Deployment Scenarios

| Scenario | Command | For |
|----------|---------|-----|
| A: macOS/Linux | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` | long-term on your machine |
| B: VPS ($5/mo) | same + `pm2 start hermes-gateway` | 24/7 |
| C: Docker | `docker pull hermes/agent:latest` | production / isolation |
| D: OpenClaw migration | `hermes claw migrate` | migrate from OpenClaw |
| E: Modal/Daytona | `hermes deploy modal` | serverless hibernation |

---

## Mode A/B: One-Liner Install

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
# installs uv + Python 3.11 (isolated), Node.js, ripgrep, ffmpeg
hermes chat "你好，今天有什么任务？"
hermes update
```

VPS daemon: `pm2 start "hermes gateway" --name hermes-gateway && pm2 save`.

## Mode C: Docker

```yaml
services:
  hermes:
    image: hermes/agent:latest
    restart: unless-stopped
    volumes:
      - ~/.hermes:/home/hermes/.hermes
      - ~/workspace:/workspace
    environment:
      - HERMES_LLM=anthropic
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
```

## Mode D: `hermes claw migrate` (from OpenClaw)

```bash
$ hermes claw migrate --dry-run   # preview first
$ hermes claw migrate             # real migration
✓ config      4 KB copied
✓ memory      38 MB copied, FTS5 index rebuilt
✓ skills      28 skills ported (3 had incompatibilities)
✓ api-keys    2 keys installed to Keyring
✓ persona     persona ported
```

### Compatibility boundary

| OpenClaw feature | Hermes status | Handling |
|------------------|---------------|----------|
| Standard Skills (`SKILL.md`) | ✅ compatible | copied |
| WhatsApp/Telegram/Discord/Slack | ✅ | copied |
| Custom TS channel adapters | ⚠️ partial | rewrite |
| WeChat / iMessage | ❌ | skipped |
| OpenClaw-only tools (TS internals) | ⚠️ mostly incompatible | auto-rewrite |

Shared-config mode (keep both running): `shared.openclaw.enabled: true, mode: read_only`.

## Mode E: Serverless (Modal / Daytona)

`hermes deploy modal` — the environment hibernates when idle and wakes on demand; long tasks keep running even when your laptop is off.

## Onboarding Wizard

```
🦤 Welcome to Hermes Agent!
? LLM provider: Nous Portal / OpenAI / Anthropic / OpenRouter / Local
? Channel: Telegram / Discord / WhatsApp / Slack / Signal / CLI
? Persona / reply language / timezone
? Self-evolving: [✓] auto skill creation [✓] offline improvement [✓] cross-session blending
```

> Enable **cross-session persona blending** — it turns on Honcho user modeling (see 15.6).

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| 5 modes | one-liner / VPS / Docker / OpenClaw migrate / serverless |
| Key commands | `hermes` / `hermes gateway` / `hermes claw migrate` / `hermes deploy` |
| Migration | `--dry-run` first; incompatible skills auto-rewritten |
| Backends | 6: local/Docker/SSH/Singularity/Modal/Daytona |

---

*Next section: [15.3 Three-Layer Architecture](./03_three_layer_architecture.md)*
