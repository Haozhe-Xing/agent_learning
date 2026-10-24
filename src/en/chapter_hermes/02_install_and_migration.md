# 13.2 Installation & Migration (from OpenClaw)

> ☤ *"From 'run it' to 'migrate from OpenClaw' — both in one command."*

---

## 5 Deployment Scenarios

| Scenario | Command | For |
|----------|---------|-----|
| A: macOS/Linux | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` | long-term on your machine |
| B: VPS (USD 5/mo) | same + `pm2 start hermes-gateway` | 24/7 |
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

Verify migration in exact source and target versions. Matching filenames, a successful command, or copied data do not establish behavioral compatibility. Use independent copies and a fixed task:

| Migration object | Verify | If acceptance fails |
|---|---|---|
| `SKILL.md` and scripts | Metadata parsing, tool dependencies, arguments, errors, and task results | Preserve originals, document differences, adapt, and retest |
| Chat channel | Identity, sessions, confirmation, and allowed actions | Keep the original connection without switching |
| Custom adapter or internal tool | Interfaces, state, paths, and permission enforcement | Do not assume automatic rewriting works; review and accept independently |
| Memory | Formats, user scope, deletion semantics, and recovery | Preserve source data and backups before deciding to switch |
| Credentials and configuration | Read, written, and ignored fields, and permission restrictions | Do not expand access or transfer unreviewed configuration |

Retain success, refusal, failure, and rollback records for identical tasks. Compatibility conclusions apply only to tested versions, configurations, and objects. A dry-run or example output does not justify deleting the source directory.

Shared-config example: `shared.openclaw.enabled: true, mode: read_only`. Before using it, test reads, writes, conflicts, and rollback on copies. Configuration alone does not prove safe shared state between two versions.

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

> The interaction above is a teaching illustration, not evidence that this exact option exists in a selected release. Check authorization, isolation, and deletion when reading [13.5, Memory](./05_memory.md) and [13.6, Proactive Tasks](./06_nudge_engine.md).

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| 5 modes | one-liner / VPS / Docker / OpenClaw migrate / serverless |
| Key commands | `hermes` / `hermes gateway` / `hermes claw migrate` / `hermes deploy` |
| Migration | Verify commands and outputs by version; test tools, state, permissions, and fixed tasks on copies |
| Backends | 6: local/Docker/SSH/Singularity/Modal/Daytona |
| Compatibility boundary | Do not infer compatibility from filenames; report tested versions and objects and preserve data for rollback |

---

*Next section: [13.3 Three-Layer Architecture](./03_three_layer_architecture.md)*
