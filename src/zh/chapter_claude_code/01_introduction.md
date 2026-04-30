# 14.1 认识 Claude Code：从零到上手

> 🖥️ *"We didn't set out to build a coding assistant. We set out to build a trusted, capable colleague who happens to work entirely in the terminal."*  
> —— Anthropic 工程团队，2024 年

---

## 从一个日常场景出发

想象你面对一个真实的工程任务：

> "我们的用户鉴权系统有个 bug，JWT token 在特定时区下会提前过期。帮我找出问题并修复，同时确保所有相关测试通过，并更新 API 文档。"

如果你使用 GitHub Copilot，你需要：
1. 自己定位问题文件，逐一打开
2. 在编辑器里手动粘贴上下文给 AI
3. 接受或拒绝每一条补全建议
4. 自己运行测试、解读结果
5. 自己更新文档

如果你使用 Claude Code，你只需在终端输入这句话——然后等待。Claude Code 会自己搜索代码库、定位问题根源、修改代码、运行测试、修复失败用例，直到全部通过为止。

**这不是"更好的代码补全"，这是 Agent 范式。**

---

## Claude Code 是什么

### 定义：CLI Agent 工具

Claude Code 是 Anthropic 推出的**命令行 AI 编程 Agent**，于 2025 年正式发布。它的本质是一个**自主行动的 AI Agent**，而非传统意义上的代码补全插件。

```bash
# 安装
npm install -g @anthropic-ai/claude-code

# 进入项目目录
cd /your/project

# 启动交互式会话
claude

# 或直接执行单次任务
claude "帮我重构 src/auth/ 目录下的鉴权逻辑，提取公共方法"
```

Claude Code 运行在你的**本地终端**，可以直接访问你的文件系统、执行 shell 命令、调用 git、读写代码文件——和一个真实的工程师坐在你旁边一样。

### Agent 范式 vs 补全范式

理解 Claude Code 的关键，在于理解它使用的是**完全不同的 AI 范式**：

![Agent范式 vs 补全范式](../svg/chapter_claude_code_01_paradigm_comparison.svg)

> 传统补全范式特点：被动响应、局部感知、单步操作  
> Agent 范式特点：主动规划、全局感知、多步操作

Claude Code 在执行任务时，内部会进行：

![Claude Code 执行流程（ReAct 循环）](../svg/chapter_claude_code_01_react_loop.svg)

这是标准的 **ReAct（Reasoning + Acting）Agent 循环**，而不是一次性的文本预测。

---

## 与传统 IDE 插件的本质区别

| 维度 | GitHub Copilot | Cursor | Claude Code |
|------|---------------|--------|-------------|
| **交互位置** | IDE 内嵌 | IDE 内嵌 | 命令行终端 |
| **工作模式** | 被动补全 | 对话+补全 | 自主 Agent |
| **感知范围** | 当前文件上下文 | 当前项目（部分） | 整个代码库 |
| **执行能力** | 仅生成文本 | 可修改文件 | 读写文件+执行命令+运行测试 |
| **任务粒度** | 函数级别 | 功能级别 | 项目级别 |
| **人类介入** | 每次补全都需确认 | 每次修改都需确认 | 可批量授权，自主完成 |
| **工具调用** | ❌ | 有限 | ✅ 完整工具链 |
| **自我验证** | ❌ | ❌ | ✅ 运行测试确认结果 |
| **定价模型** | 按订阅 | 按订阅 | 按 Token 使用量 |

### 一个具体对比

**任务**：*"在所有 API 端点中添加请求速率限制，限制每个用户每分钟最多 100 次调用。"*

**GitHub Copilot 的做法**：
- 你需要打开每一个路由文件
- 告诉 Copilot 当前上下文
- 接受它生成的速率限制代码片段
- 手动检查是否所有端点都覆盖到了
- 手动运行测试

**Claude Code 的做法**：
```bash
$ claude "在所有 API 端点添加速率限制，每用户每分钟 100 次，使用 Redis 做计数器"

✓ 分析项目结构，发现 23 个 API 路由
✓ 找到现有的 middleware 模式
✓ 创建 rate_limiter.py（Redis 实现）
✓ 修改 23 个路由文件，注入中间件
✓ 更新单元测试，运行全部测试（47/47 通过）
✓ 更新 API 文档中的速率限制说明

完成。共修改 25 个文件，所有测试通过。
```

---

## Claude Code 的设计哲学

### 1. Unix 工具哲学：做好一件事

Claude Code 坚守 Unix 工具传统：**一个工具，专注于一个核心功能，通过组合实现强大能力**。

它不尝试成为全功能的 IDE，不内嵌调试器，不提供图形界面。它只做一件事：**在终端里作为可信赖的 AI 编程助手帮你完成工程任务**。

这种克制，带来了极佳的**可组合性**：

```bash
# 与 git hooks 组合
echo 'claude "检查这次提交是否有潜在的安全问题"' > .git/hooks/pre-commit

# 与 CI/CD 组合
claude "分析这次 build 失败的原因并修复" --pipe < build_log.txt

# 与其他 CLI 工具管道组合
git diff HEAD~1 | claude "总结这个 PR 的改动，生成 changelog 条目"
```

### 2. 可信任：透明而非黑盒

Claude Code 设计上强调**操作透明性**。每一步操作，它都会明确告知用户：

```
Claude Code 的默认行为
────────────────────
✓ 读取文件：直接执行，会展示读取了哪些文件
⚠️ 修改文件：默认会请求确认（可配置）
⚠️ 执行命令：默认会展示命令并请求确认
❌ 危险操作：删除文件、git push 等，始终请求确认
```

用户可以通过权限配置精确控制 Claude Code 的行为范围：

```json
// .claude/settings.json
{
  "permissions": {
    "allow": [
      "Bash(git:*)",         // 允许所有 git 操作
      "Read(**/*.py)",        // 允许读取所有 Python 文件
      "Edit(src/**/*)"        // 允许修改 src 目录下的文件
    ],
    "deny": [
      "Bash(rm:*)",           // 禁止删除操作
      "Bash(curl:*)"          // 禁止网络请求
    ]
  }
}
```

### 3. 尽量少做：最小权限，最小副作用

> 🎯 *"The best action is the one that accomplishes the goal with the least irreversible side effects."*  
> —— Anthropic, Claude Code 设计文档

Claude Code 遵循**最小必要操作原则**：

- 只读取任务所需的文件，不扫描整个磁盘
- 只修改需要修改的部分，不进行"顺便重构"
- 遇到歧义时，询问用户而不是自行猜测
- 不主动执行有副作用的操作（网络请求、数据库写入等）

这与某些"激进 Agent"设计形成对比——后者倾向于"尽可能多做"，往往带来不可预期的副作用。

---

## 安装与初始化

### 环境要求

```bash
# Node.js 18+
node --version  # v18.0.0 或以上

# 需要 Anthropic API Key
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 安装

```bash
# 通过 npm 全局安装
npm install -g @anthropic-ai/claude-code

# 验证安装
claude --version
# claude v1.x.x
```

### 初始化项目

```bash
# 进入你的项目
cd /path/to/your/project

# 启动 Claude Code（首次会初始化 .claude/ 配置目录）
claude

# 首次启动界面示意：
# ╔══════════════════════════════════════╗
# ║  Claude Code v1.x.x                  ║
# ║  Working in: /path/to/your/project   ║
# ║                                      ║
# ║  Type /help for commands             ║
# ╚══════════════════════════════════════╝
# >
```

### 项目记忆：CLAUDE.md

Claude Code 的核心配置机制是 `CLAUDE.md` 文件——它是 Claude Code 的"项目说明书"：

```markdown
# CLAUDE.md（放在项目根目录）

## 项目概述
这是一个 FastAPI + PostgreSQL 的用户管理系统。

## 技术栈
- Python 3.11, FastAPI 0.100+
- PostgreSQL 14, SQLAlchemy 2.0
- pytest 用于测试

## 代码规范
- 所有函数必须有类型注解
- 使用 Black 格式化，行宽 88 字符
- 新功能必须附带单元测试

## 禁止事项
- 不要修改 migrations/ 目录下的已有文件
- 不要直接操作数据库，通过 Repository 层
```

这个文件会在每次会话开始时被自动读取，确保 Claude Code 始终了解项目的上下文和约束。

---

## 第一个实际案例演示

让我们通过一个完整的真实案例，感受 Claude Code 的工作方式。

### 场景：修复一个生产 Bug

**项目**：一个 Python Flask 应用，有用户报告说"搜索功能在输入特殊字符时会崩溃"。

```bash
$ cd flask-app
$ claude
```

**用户输入**：
```
> 用户反馈搜索功能崩溃，说输入了特殊字符。帮我找出问题并修复，确保测试通过
```

**Claude Code 的完整执行过程**：

```
🔍 分析任务：查找搜索功能的 bug，涉及特殊字符处理

📂 探索代码库结构...
  读取 app.py
  读取 routes/search.py
  读取 tests/test_search.py

💡 发现问题：
  routes/search.py:47 中，用户输入直接拼接到 SQL 查询中
  存在 SQL 注入漏洞，特殊字符（如单引号）导致查询语句语法错误

📋 修复计划：
  1. 将原始字符串拼接改为参数化查询
  2. 添加输入验证层
  3. 补充边界情况的测试用例

⚙️ 执行修改...
```

**修改前（有问题的代码）**：
```python
# routes/search.py（修改前）
@app.route('/search')
def search():
    query = request.args.get('q', '')
    # ⚠️ 危险：直接字符串拼接
    sql = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
    results = db.execute(sql)
    return jsonify(results)
```

**修改后（Claude Code 的修复）**：
```python
# routes/search.py（修改后）
@app.route('/search')
def search():
    query = request.args.get('q', '')
    
    # ✅ 输入验证
    if len(query) > 200:
        return jsonify({"error": "搜索词过长"}), 400
    
    # ✅ 参数化查询，防止 SQL 注入
    sql = "SELECT * FROM products WHERE name LIKE :pattern"
    results = db.execute(sql, {"pattern": f"%{query}%"})
    return jsonify([r._asdict() for r in results])
```

**Claude Code 继续执行**：
```
✅ 修改完成，运行测试...

$ pytest tests/test_search.py -v
  PASSED test_search_normal_query
  PASSED test_search_empty_query  
  FAILED test_search_special_chars  ← 之前没有这个测试，现在 Claude 补充了

正在添加特殊字符测试用例...
```

**Claude Code 添加的新测试**：
```python
# tests/test_search.py（新增部分）
def test_search_special_chars():
    """测试特殊字符不会导致崩溃"""
    special_inputs = ["'", '"', "'; DROP TABLE products; --", "%", "_", "\\"]
    
    for input_str in special_inputs:
        response = client.get(f'/search?q={quote(input_str)}')
        assert response.status_code == 200, f"特殊字符 '{input_str}' 导致崩溃"
```

**最终输出**：
```
✅ 所有测试通过（12/12）
✅ 修复了 SQL 注入漏洞
✅ 添加了 6 个特殊字符边界测试

修改文件：
  - routes/search.py（参数化查询）
  - tests/test_search.py（新增测试用例）

建议：还发现 routes/user.py 中存在类似的 SQL 拼接问题，
是否也一并修复？
```

整个过程**不需要用户手动定位问题、不需要手动修改代码、不需要手动运行测试**——Claude Code 自主完成了从"发现问题"到"验证修复"的完整闭环。

---

## 核心能力概览

Claude Code 的能力覆盖软件开发的全生命周期：

### 代码理解与分析

```bash
# 解释复杂代码
> 解释一下 src/core/scheduler.py 中的任务调度算法

# 代码审查
> 审查最近 3 个 commit 的代码，找出潜在问题

# 架构分析
> 描述这个项目的整体架构，画出模块依赖关系
```

### 代码编写与修改

```bash
# 实现新功能
> 实现一个用户权限管理系统，支持 RBAC 模型

# 重构代码
> 将 utils.py 中的所有辅助函数按功能分类，拆分成独立模块

# Bug 修复
> tests/ 目录下有 3 个 failing test，找出原因并修复
```

### 测试与质量

```bash
# 生成测试
> 为 src/payment/ 目录下所有模块生成单元测试，覆盖率达到 80%

# 运行测试并修复
> 运行全量测试，如果有失败的，自动修复直到全部通过

# 代码质量检查
> 运行 pylint，修复所有 E 级别的错误
```

### Git 与工程流程

```bash
# 智能提交
> 将当前改动整理成一个合适的 commit，生成规范的 commit message

# PR 描述
> 基于这个分支的改动，生成详细的 PR description

# Changelog 生成
> 根据从 v1.2.0 到现在的 git log，生成 CHANGELOG.md
```

### 文档与沟通

```bash
# API 文档
> 为 routes/ 目录下所有端点生成 OpenAPI 文档注释

# 技术文档
> 为这个模块写一个 README，包含安装、使用示例和 API 说明

# 代码注释
> 为 src/algorithms/ 下的复杂算法添加详细的行内注释
```

### 能力边界速览

```
✅ Claude Code 擅长的：
  · 跨文件的代码理解和修改
  · 遵循已有代码风格
  · 运行命令并根据输出调整
  · 迭代修复直到测试通过
  · 发现潜在问题并主动提示

⚠️ Claude Code 需要人类协助的：
  · 产品需求决策（"应该实现哪个功能"）
  · 架构级别的重大决策
  · 涉及业务逻辑的歧义判断
  · 需要外部系统权限的操作

❌ Claude Code 不做的：
  · 未经授权修改生产数据库
  · 绕过用户设置的权限限制
  · 在没有确认的情况下执行高风险操作
```

---

## 本节小结

| 概念 | 要点 |
|------|------|
| **Claude Code 定位** | CLI Agent 工具，而非 IDE 插件 |
| **核心范式** | Agent（自主规划执行）而非补全（被动预测） |
| **与 Copilot/Cursor 的区别** | 全局感知、自主执行、自我验证 |
| **设计哲学** | Unix 风格（专注）、透明（可信任）、最小副作用（尽量少做） |
| **核心配置** | CLAUDE.md 提供项目上下文，settings.json 控制权限 |
| **能力范围** | 代码理解→编写→测试→文档的完整开发循环 |

> 💡 **关键认知转变**：使用 Claude Code 时，你不再是"写代码的人"，而是"描述目标、审核结果的人"。这需要你将思维从"怎么写这段代码"转变为"我希望这个系统实现什么行为"。

---

## 参考资料

[1] ANTHROPIC. Claude Code: Deep dive into agentic coding[EB/OL]. Anthropic Blog, 2025.

[2] ANTHROPIC. Claude Code documentation[EB/OL]. (2025)[2026-04-07]. https://docs.anthropic.com/claude-code.

[3] ANTHROPIC. Building effective agents[EB/OL]. Anthropic Blog, 2024-12.

[4] HOGAN B. The pragmatic programmer: your journey to mastery[M]. 20th Anniversary Edition. The Pragmatic Bookshelf, 2019.

---

*下一节：[15.2 Claude Code 的工具系统](./02_tools.md)*
