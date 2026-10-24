# 15.2 安装与快速上手（含 OpenClaw 迁移）

> ☤ *"从'跑起来'到'用 OpenClaw 数据迁过来'，都在一行命令里。"*

---

## 一、5 种部署场景

Hermes Agent 给用户的部署选项和 OpenClaw 类似，但**多了"从 OpenClaw 迁移"的快捷路径**。下面是一张 5 种场景的对照表（每种都给一行命令）：

| 场景 | 命令 | 适合 |
|------|------|------|
| **A: 个人 macOS / Linux** | `curl -fsSL https://hermes-agent.nousresearch.com/install.sh \| bash` | 长期在自己机器上 |
| **B: VPS（$5/月起的 Linux）** | 同 A + `pm2 start hermes-gateway` | 想 24/7 在线 |
| **C: Docker 沙箱** | `docker pull hermes/agent:latest` | 生产化、隔离 |
| **D: OpenClaw 迁移** | `hermes claw migrate` | 从 OpenClaw 平迁 |
| **E: Modal/Daytona（Serverless 休眠）** | `hermes deploy modal` | 不想开机就能用 |

下面按 A→E 顺序展开。

---

## 二、方式 A：个人 macOS / Linux 一行安装

### 2.1 一行命令

```bash
$ curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

脚本会：

1. 检测是否已安装 `uv`（Hermes 自带 Python 3.11 + uv 隔离环境，独立于系统 Python）；
2. 安装 Node.js（渠道网关需要）；
3. 安装 ripgrep（记忆的全文检索需要）；
4. 安装 ffmpeg（语音备忘录转写需要）；
5. 启动 `hermes setup`（首次让你配置 LLM provider + 至少一个渠道）；
6. 完成后输出"Start chatting: hermes"。

### 2.2 验证

```bash
$ hermes --version
hermes 0.x.x       # 实际版本以仓库 release 为准

$ hermes chat "你好，今天有什么任务？"
🦤 Hermes Agent here. Based on our past conversations, today is …
```

### 2.3 升级

```bash
$ hermes update
```

升级会同步更新核心代理、技能市场、网关服务。

---

## 三、方式 B：VPS 上 24/7 守护

适用于把 Hermes 放在 VPS 上长期运行。基础步骤和 A 相同，但额外需要进程守护。

```bash
# 1. 安装（同 A）
$ curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

# 2. 启动 gateway
$ pm2 start "hermes gateway" --name hermes-gateway

# 3. 守护开机自启
$ pm2 save
$ pm2 startup

# 4. 反向代理（如果用 Web UI）
$ pm2 start "hermes web" --name hermes-web
```

`$5/月` 的 VPS 足够跑——CPU 跑 LLM 调用（用云端 API）、存储 25 GB 足够、技能市场增量更新。

---

## 四、方式 C：Docker 沙箱部署

如果你不想让 Hermes 直接触达你的文件系统，可以走 Docker：

```bash
$ docker pull hermes/agent:latest

$ docker run -d --name hermes \
    -v ~/.hermes:/home/hermes/.hermes \      # 配置 + 记忆持久化
    -v ~/workspace:/workspace \               # Agent 可访问的工作区
    -e HERMES_LLM=anthropic \
    -e ANTHROPIC_API_KEY=sk-... \
    -p 7777:7777 \                            # TUI / Web 端口
    hermes/agent:latest gateway
```

`docker-compose.yml`（生产推荐）：

```yaml
version: "3.9"

services:
  hermes:
    image: hermes/agent:latest
    container_name: hermes
    restart: unless-stopped
    volumes:
      - ~/.hermes:/home/hermes/.hermes
      - ~/workspace:/workspace
    environment:
      - HERMES_LLM=anthropic
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - HERMES_NETWORK=allowlist  # 默认仅白名单域名
    ports:
      - "7777:7777"
```

> 容器化版本对 Hermes 的"自动创建 Skill + 离线迭代"功能**完全**支持——所有 skill 文件落盘到 `~/.hermes/skills/`，不会因容器重启丢失。

---

## 五、方式 D：从 OpenClaw 平迁（`hermes claw migrate`）

这是 Hermes 给 OpenClaw 用户的最贴心的功能。一行命令，把 OpenClaw 数据全套迁过来。

### 5.1 准备

在迁之前，确保：

- OpenClaw 已经在某处运行过（`~/.openclaw/` 存在）；
- Hermes 已经按方式 A/B/C 装好；
- 当前用户对两个目录都有读权限。

### 5.2 试运行（推荐）

```bash
$ hermes claw migrate --dry-run

? Plan:
  source: /Users/you/.openclaw
  target: /Users/you/.hermes

  items:
    config.yaml                     → ~/.hermes/config.yaml              [4 KB]
    memory/db.sqlite                → ~/.hermes/memory/main.sqlite       [38 MB]
    skills/        (28 dirs)        → ~/.hermes/skills/                  [12 MB]
    api-keys/                       → Keyring + Hermes credential store  [2 KB]
    persona/                        → ~/.hermes/persona/                 [2 MB]

? Proceed?  [y/N]
```

`--dry-run` 只打计划，不动文件。建议先跑一遍。

### 5.3 真正迁移

```bash
$ hermes claw migrate
```

执行期间你会看到：

```
✓ config           4 KB copied
✓ memory           38 MB copied, FTS5 index rebuilt
✓ skills           28 skills ported (3 had incompatibilities, see below)
✓ api-keys         2 keys installed to Keyring
✓ persona          persona "工作模式 · 中文" ported

⚠ Compatibility notes:
  - skill "send-imessage"   : iMessage not supported by Hermes (channel not in 15+ providers)
  - skill "ocr-image"       : backend lib differs; auto-rewritten to use Hermes' generic OCR plugin
  - skill "openclaw-native-fs": Hermes uses sandboxed fs plugin; auto-mapped
```

迁移完成后：

```bash
$ hermes doctor
✓ Memory: 38 MB, indexed
✓ Skills: 25 ready, 3 rewritten
✓ Channels: Telegram bot online
```

### 5.4 兼容性边界（**重要**）

Hermes 是 OpenClaw 的"扩展演化"，不是"完全超集"。迁移时会遇到若干兼容性问题，常见的有：

| OpenClaw 特性 | Hermes 兼容状态 | 处理 |
|---------------|---------------|------|
| 标准 Skills（`SKILL.md` + 脚本） | ✅ 完全兼容 | 直接复制 |
| WhatsApp / Telegram / Discord / Slack | ✅ 完全支持 | 直接复制 |
| `SKILL.md` frontmatter 字段 | ✅ 兼容 | 直接复制 |
| 自定义 TypeScript 渠道适配器 | ⚠️ 部分兼容 | 需要重写 |
| 微信 / iMessage 等闭源渠道 | ❌ 不支持 | 跳过 |
| 自定义 OpenClaw-only Tools（TS 内部 API） | ⚠️ 大部分不兼容 | 自动重写 |
| `agent_memory.db`（OpenClaw 长期记忆格式） | ⚠️ 转换 | 自动转换 |

迁移完后，建议先跑一周 `hermes claw migrate --dry-run` 对比的副本来确保一致；然后再删 OpenClaw。

### 5.5 共享配置文件

如果你想保留 OpenClaw 不删，Hermes 提供了 `shared_config` 模式：

```yaml
# ~/.hermes/config.yaml
shared:
  openclaw:
    enabled: true
    path: /Users/you/.openclaw
    mode: read_only   # Hermes 读 OpenClaw 的数据，但写回自己的目录
```

这是开发期间非常实用的模式——你能同时用 OpenClaw 和 Hermes 比对行为。

---

## 六、方式 E：Serverless 后端（Modal / Daytona）

Hermes 的两个高级执行后端是：

- **Modal** —— Serverless Python 平台，环境**平时休眠**，按需唤醒；
- **Daytona** —— 类似但更倾向"开发沙箱"。

这两个后端的共同好处是：

```
Hermes 在对话中说："这个任务需要跑 5 分钟的批处理"
         │
         ▼
   Modal/Daytona 自动启动容器执行
         │
         ▼
   执行期间计费（按秒/按 token）
         │
         ▼
   完成后自动休眠，下次调用再启动
```

这种模式的好处是：**Hermes 在你电脑关机时仍能执行长任务**。

```bash
# 一键部署到 Modal
$ hermes deploy modal --name my-hermes

# 部署到 Daytona（开发沙箱模式）
$ hermes deploy daytona
```

> Modal / Daytona 后端 **会**按调用时长和资源消耗收费。具体计费规则以各自平台为准；本书不引用具体价格。

---

## 七、首次启动：Onboarding Wizard

无论选哪种部署方式，首次启动都会进入同一个 onboarding 向导：

```
🦤 Welcome to Hermes Agent!

?  Pick an LLM provider:
  ( ) Nous Portal     (recommended; one key covers model + web search + vision)
  ( ) OpenAI          (need OpenAI API key)
  ( ) Anthropic       (need Anthropic API key)
  ( ) OpenRouter      (one key, many models)
  ( ) Local           (Ollama / llama.cpp / vLLM)

?  Pick at least one channel:
  [x] Telegram         BotFather token
  [ ] Discord          Bot token
  [ ] WhatsApp         QR scan
  [ ] Slack            App token
  [ ] Signal           signal-cli daemon
  [x] CLI              (always on)

?  Personalization:
  Persona name: ____
  Reply in: [Chinese | English | both]
  Timezone: Asia/Shanghai

?  Self-evolving skills:
  [✓] Enable automatic skill creation
  [✓] Enable offline skill improvement
  [ ] Enable cross-session persona blending

✓ Setup complete. Try: hermes chat
```

`✓ Enable cross-session persona blending` 是**首次启动强烈建议开**——它给 Hermes 打开用户建模（Honcho dialectic）的开关，后面 15.6 会详细讲。

---

## 八、本节小结

| 主题 | 关键要点 |
|------|---------|
| 5 种部署 | 一行安装 / VPS / Docker / OpenClaw 迁移 / Serverless |
| 关键命令 | `hermes` / `hermes gateway` / `hermes claw migrate` / `hermes deploy <backend>` |
| OpenClaw 迁移 | `hermes claw migrate [--dry-run]`；处理 Skills 不兼容时会自动重写 |
| 后端选择 | 6 种：local / Docker / SSH / Singularity / Modal / Daytona |
| 兼容性边界 | 标准 Skill 完全兼容；TS 内部 API / 闭源渠道不兼容 |

---

*下一节：[15.3 三层架构：Gateway / Engine / Plugin 子系统](./03_three_layer_architecture.md)*
