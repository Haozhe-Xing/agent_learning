# 17.2 Installation & 4 Run Modes

> 🐋 *"One command starts an Agent; four profiles decide its capability boundary."*

---

## Four Profiles at a Glance

DeepSeek Harness uses **profile-based startup** — `dsh --profile <name>` loads a preset plugin set:

| Profile | Name | For | Plugin set |
|---------|------|-----|------------|
| **standard** | standard | daily Agent dev / production | full tools + sandbox + scheduler + TUI |
| **minimal** | minimal | model benchmarking | shell + file edit only |
| **ptc** | PTC | long-chained tool calling | tools + PTC adapter + workflow |
| **create** | create | plugin prototyping | all plugins + creation workflow |

---

## Mode A: standard — Daily Work

```bash
npx @deepseek-ai/dsh web          # Web UI at 127.0.0.1:3080
dsh chat                          # CLI
dsh gateway                       # Telegram/Discord etc.
```

Config (`~/.dsh/config.json`):

```json
{
  "profile": "standard",
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-v4-pro",
    "apiKey": "${DEEPSEEK_API_KEY}",
    "fallback": ["anthropic-claude-4-7", "openai-gpt-4.1"]
  },
  "workspace": "/workspace",
  "mcpServers": { "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"]} },
  "permissions": {
    "shell": ["ls","cat","grep","find","pwd"],
    "shellBlocklist": ["rm -rf /","sudo","chmod 777","curl | sh"]
  }
}
```

## Mode B: minimal — Benchmarking

```bash
dsh --profile minimal eval --benchmark swe-bench
```

Only shell + file edit — controls for variables so model capability itself is measured, not tool misuse. Any paper using DeepSeek Harness should state the profile explicitly.

## Mode C: ptc — Programmatic Tool Calling

PTC lets the LLM **generate a program** (Python/TS) that the harness executes to call tools — N tool calls collapse into one LLM output:

```
Normal loop:  LLM → tool_call(read) → execute → result → LLM → tool_call(grep) → ...
PTC:          LLM outputs Python (read files, find bugs, edit) → harness executes all
```

Pros: token savings, readability, composability (conditionals/loops/try-except). Cons: needs stricter sandbox (PTC sandbox blocks outbound network, non-workspace writes, `subprocess`, `os.system` by default).

## Mode D: create — Plugin Prototyping

```bash
dsh --profile create dev
```

Watches `./plugins/**/*.ts`; auto-reloads in ~200ms on save — sub-second feedback loop for plugin development.

## Install: 3 Ways

```bash
# 1) npx
npx @deepseek-ai/dsh web

# 2) source
git clone https://github.com/deepseek-ai/deepseek-harness.git
cd deepseek-harness && corepack enable && pnpm install && pnpm dsh web

# 3) Docker
docker run -d --name dsh -v ~/.dsh:/root/.dsh -v ~/workspace:/workspace \
    -e DEEPSEEK_API_KEY=sk-... -p 3080:3080 deepseek/harness:latest web
```

Repo layout (per `main`): `packages/` (core/llm/shell/terminal/fs/edit/lsp/web/skill/subagent/workflow/mcp/scheduler/...), `plugins/`, `skills/`, `profile/` (4 JSON), `docs/`.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| 4 profiles | standard / minimal / ptc / create |
| Decision | daily / benchmark / long-chain / plugin-dev |
| Install | npx / source / Docker |
| PTC sandbox | stricter than standard |
| create | hot-reload, sub-second loop |

---

*Next section: [17.3 Architecture: Cordis Microkernel & Plugin Topology](./03_cordis_architecture.md)*
