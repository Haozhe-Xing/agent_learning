# 17.2 安装与四种运行模式

> 🐋 *"一行命令启动一个 Agent，但四种姿势决定 Agent 的能力边界。"*

---

## 一、四种姿势一览

DeepSeek Harness 的核心设计之一是 **profile-based 启动**——根据你想要的能力组合，选不同 profile。每次启动时 `dsh --profile <name>`，Harness 会自动加载对应的插件集合：

| Profile | 中文名 | 适用场景 | 核心插件组合 |
|---------|--------|---------|-------------|
| **standard** | 标准模式 | 日常 Agent 开发 / 生产 | 全套工具 + 沙箱 + 调度 + TUI |
| **minimal** | 极简模式 | 模型基准测试 / 性能对比 | 仅 shell + 文件编辑 |
| **ptc** | PTC 模式 | 长链式工具调用 / 程序化工作流 | 工具 + PTC 适配器 + 工作流 |
| **create** | 创造模式 | 在内存里实验 / 自定义插件原型 | 全部插件 + 插件创作工作流 |

下面按模式展开，每种都给一行启动命令 + 适用人群。

---

## 二、方式 A：standard（标准模式）——日常干活

### 2.1 一行命令

```bash
$ npx @deepseek-ai/dsh web
```

启动 Web UI（默认 127.0.0.1:3080）；`dsh chat` 进入 CLI 交互；`dsh gateway` 启动 Telegram/Discord 等接入。

### 2.2 加载的插件

```
core.llm.openai
core.llm.anthropic
core.tool.shell
core.tool.fs
core.tool.edit
core.tool.web
core.tool.lsp
core.skill.loader
core.context.summary
core.agent.loop
core.subagent.spawn
core.session.sqlite
core.scheduler.cron
core.storage.local
core.sandbox.local
core.tui.ink
core.web.react          (启 --web 时)
```

约 50~70 个插件组成**日常 Agent 的标准工具箱**。

### 2.3 配置文件

```json
// ~/.dsh/config.json —— standard profile 默认加载
{
  "profile": "standard",
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-v4-pro",
    "apiKey": "${DEEPSEEK_API_KEY}",
    "fallback": ["anthropic-claude-4-7", "openai-gpt-4.1"]
  },
  "workspace": "/workspace",
  "mcpServers": {
    "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] },
    "postgres": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-postgres"] }
  },
  "skills": ["./skills/my-team.md"],
  "permissions": {
    "shell": ["ls", "cat", "grep", "find", "pwd"],
    "shellBlocklist": ["rm -rf /", "sudo", "chmod 777", "curl | sh"]
  }
}
```

`apiKey: "${DEEPSEEK_API_KEY}"` 走环境变量——避免硬编码。

### 2.4 第一条指令

```bash
$ dsh chat "读 idea 文档, 理解愿景、坚持原则。先梳理要设计的条目，及其组织、优先级。"
```

> 💡 这条指令是 DeepSeek 团队在 README 里给的入门示例——典型的"开放域 + 多步任务"。

---

## 三、方式 B：minimal（极简模式）——跑基准测试

### 3.1 一行命令

```bash
$ dsh --profile minimal eval --benchmark swe-bench
```

只保留 **shell + 文件编辑** 两种工具 + 模型调用——用于：

- 模型基准（不给工具 = 比纯推理；给最基础工具 = 比基本 Agent 能力）；
- 学术论文的 control group；
- 评测 A/B 测试。

### 3.2 它解决了什么

在 standard 模式里跑基准测试会被"工具太多"污染——Agent 可能**误用工具**绕过对模型能力本身的评估。minimal profile 把变量降到极致：

```
可用工具：shell, fs.read, fs.write
不可用：web, lsp, subagent, scheduler, skill, mcp
```

这相当于"裸 LLM + 极简工具集"。

### 3.3 学术引用场景

如果未来你想引用 DeepSeek Harness 做研究实验，**minimal profile 是事实上的标准**——任何论文应该明确说明 profile，避免"配置不一致导致结论不可比"。

---

## 四、方式 C：ptc（PTC 模式）——程序化工具调用

### 4.1 什么是 PTC

**PTC（Programmatic Tool Calling）** 模式让 LLM **先生成一段程序**（Python 或 TypeScript），再由 Harness 执行这段程序调用工具。

```
普通 Agent Loop（每步一工具调用）：
  LLM → tool_call(read) → execute → result
  LLM → tool_call(grep) → execute → result
  LLM → tool_call(edit) → execute → result

PTC 模式（程序化执行）：
  LLM 输出 Python：
    import read, grep, edit
    files = read("./src/**/*.py")
    bugs = [f for f in files if "TODO" in f.content]
    for f in bugs[:5]:
        edit(f, replace="...")
    return f"修复 {len(bugs)} 个 TODO"
  Harness 执行 → 完成全部
```

**优势**：

1. **省 token**：N 个 tool call 在 PTC 模式下合并成 1 次 LLM 输出；
2. **可读性**：代码比结构化 JSON 字典更像人话表达；
3. **可组合**：条件、循环、try/except 都自然存在。

**劣势**：

1. **安全**：执行任意代码需要更严格的沙箱；
2. **调试**：错误时栈帧更深；
3. **能力**：模型要会"写程序"。

### 4.2 启动

```bash
$ dsh --profile ptc chat "读 src/auth/*.py，找出所有未处理的异常"
```

PTC profile 默认加载：

```
core.llm.openai
core.llm.anthropic
core.tool.shell
core.tool.fs
core.tool.edit
ptc.python.interpreter
ptc.sandbox.restricted   ← 比标准沙箱更严
core.context.summary
```

### 4.3 安全沙箱

PTC 沙箱**默认**拒绝：

- 网络出站（除白名单）
- 文件系统写（除 `/workspace/`）
- 进程派生（除白名单）
- 导入危险模块（`subprocess`、`os.system` 等）

如果想让 PTC 更宽松，编辑 `~/.dsh/profiles/ptc.json` 把沙箱调到 `dev` 等级。

---

## 五、方式 D：create（创造模式）——写插件原型

### 5.1 一行命令

```bash
$ dsh --profile create dev
```

启动一个**带 hot-reload 的开发环境**——你写一个插件保存，Harness 即时加载新插件，**无需重启**。

### 5.2 它解决了什么

写插件时最痛的环节是"改一行插件 → 重启 Harness → 测试 → 重启 → …"。create profile 把这个循环缩到 < 1 秒：

```
$ dsh --profile create dev
   → Boots with create profile
   → Watches ./plugins/**/*.ts for changes
   → On save: auto-reload plugin in 200ms
   → Logs to console with colored diff
```

### 5.3 装载 demo

create profile 默认装载 `example.plugins`——包含以下示范插件，你可以基于它们写自己的：

- `demo.tool.weather` —— 调 OpenWeatherMap；
- `demo.llm.placeholder` —— 一个把"输入直接返回"的假模型；
- `demo.skill.hello` —— 一个简单的 `SKILL.md` Skill。

学习插件 API 的最快方法是修改 `example.plugins`——改完保存，立刻看到效果。

---

## 六、Pre-4 模式选择：决策树

```
Q1: 你想跑什么场景？
├─ 日常 Agent 开发 / 干活 → standard
├─ 模型基准 / 学术对比 → minimal
├─ 多步任务 / 长链工作流 → ptc
└─ 写自定义插件 → create

Q2: 第一次上手？
└─ 用 standard，它装载最完整的插件组合
```

---

## 七、安装三种方式（含 Docker / 源码）

### 7.1 方式 1：`npx` 一行启动

```bash
$ npx @deepseek-ai/dsh web
```

适用：试用 / demo。

### 7.2 方式 2：源码构建（推荐开发）

```bash
$ git clone https://github.com/deepseek-ai/deepseek-harness.git
$ cd deepseek-harness
$ corepack enable
$ pnpm install
$ pnpm dsh web
```

仓库结构（**以 `main` 分支为准**）：

```
deepseek-harness/
├── packages/
│   ├── core/                # Cordis 集成 + 上下文键
│   ├── llm/                 # 各 LLM Provider
│   ├── shell/               # shell + sandbox
│   ├── terminal/            # TUI（终端 UI）
│   ├── fs/                  # 文件系统操作
│   ├── edit/                # 编辑工具
│   ├── lsp/                 # LSP 集成
│   ├── web/                 # Web 搜索
│   ├── skill/               # Skill 加载器
│   ├── subagent/            # 子 Agent
│   ├── workflow/            # 工作流
│   ├── mcp/                 # MCP server / client
│   ├── scheduler/           # cron 等
│   └── ...
├── plugins/                 # 社区插件
├── skills/                  # 默认 Skills
├── profile/                 # 4 种 profile 配置
│   ├── standard.json
│   ├── minimal.json
│   ├── ptc.json
│   └── create.json
└── docs/                    # 文档源
```

> 子目录的具体命名以仓库 `main` 分支为准。

### 7.3 方式 3：Docker 沙箱

```bash
$ docker pull deepseek/harness:latest

$ docker run -d --name dsh \
    -v ~/.dsh:/root/.dsh \                # 配置 + 持久化
    -v ~/workspace:/workspace \           # 工作区
    -e DEEPSEEK_API_KEY=sk-... \
    -p 3080:3080 \                        # Web UI 端口
    deepseek/harness:latest web
```

Web UI 跑在 `127.0.0.1:3080`。

---

## 八、DeepSeek API 的峰谷定价

DeepSeek 在 2026 年 8 月起对 API 价格做了调整——区分峰时段与谷时段。**长任务可考虑在谷时段跑**以节省成本。具体价格以 `platform.deepseek.com` 官方公示为准；本书不引用具体数字。

要在 Harness 里强制使用谷时段路由：

```json
{
  "llm": {
    "provider": "deepseek",
    "model": "deepseek-v4-pro",
    "routingPolicy": "off-peak-preferred"
  }
}
```

---

## 九、本节小结

| 主题 | 关键要点 |
|------|---------|
| 4 种 Profile | standard / minimal / ptc / create |
| 选型决策 | 日常 / 基准 / 长链 / 写插件 |
| 安装方式 | npx / 源码 / Docker |
| PTC 沙箱 | 比 standard 严格；可调到 dev |
| create 模式 | hot-reload，秒级反馈循环 |
| DeepSeek API | 峰谷定价，长任务可走谷时段 |

---

*下一节：[17.3 架构：Cordis 微内核与插件拓扑](./03_cordis_architecture.md)*
