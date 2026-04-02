# 9.3 AGENTS.md / CLAUDE.md：Agent 宪法写作指南

> 📜 *"代码仓库是唯一的真相源（Source of Truth）。所有指导 AI Agent 行为的知识，必须以机器可读的形式存储在代码库中。"*  
> —— OpenAI Codex 团队工程实践

---

## 什么是 AGENTS.md / CLAUDE.md？

在 Harness Engineering 中，**`AGENTS.md`**（或 `CLAUDE.md`）是放在项目根目录的一个特殊文件，被称为 **Agent 宪法**（Agent Constitution）。

它的作用是：**在 Agent 开始工作之前，告诉它这个项目的规则、架构、约定和边界。**

```
项目根目录/
├── AGENTS.md          ← Agent 宪法（通用，适用于所有 AI Agent）
├── CLAUDE.md          ← Claude 专用版本（由 Claude Code 自动读取）
├── src/
├── tests/
└── ...
```

> **为什么需要它？**  
> 每次 Agent 开始新任务时，它都是"失忆"的——不知道这个项目的架构决策、代码风格、禁止操作……  
> 如果不告诉它，它会用"通用最佳实践"来推断，这往往与你的项目约定不符。

---

## Agent 宪法的三大核心原则

在开始写内容之前，先了解三个核心原则：

### 原则一：机器可读，而非人类可读

```markdown
# ❌ 糟糕的写法（面向人类的叙述性文字）
这个项目使用 Python，建议你在写代码时注意代码质量，
保持良好的命名习惯，尽量写清楚注释，这样其他同事
读起来也会更方便...

# ✅ 好的写法（结构化，机器可读）
## 语言与框架
- 语言：Python 3.11+
- Web 框架：FastAPI
- ORM：SQLAlchemy 2.0
- 测试：pytest + pytest-asyncio

## 代码规范
- 命名：蛇形命名（snake_case），类名使用大驼峰（PascalCase）
- 类型提示：所有公共函数必须有完整类型提示
- 文档字符串：使用 Google 风格
```

### 原则二：渐进式披露，而非大而全

OpenAI 团队发现：一个 5000 行的 `AGENTS.md` 反而比一个 200 行的 `AGENTS.md` 效果更差。原因是 Agent 会在大量信息中"迷失"。

**正确做法**：`AGENTS.md` 是"目录"，详细内容在独立文件中：

```markdown
## 架构概览
详见 [docs/architecture.md](./docs/architecture.md)

## API 规范
详见 [docs/api-conventions.md](./docs/api-conventions.md)

## 测试指南  
详见 [docs/testing-guide.md](./docs/testing-guide.md)
```

### 原则三：规范行为，而非描述状态

```markdown
# ❌ 描述性（告诉 Agent 现在是什么状态）
我们使用 PostgreSQL 数据库。

# ✅ 行为性（告诉 Agent 该怎么做）
修改数据库 Schema 时：
1. 必须创建 Alembic migration 文件，不得直接修改模型
2. Migration 文件名格式：{timestamp}_{description}.py
3. 提交前运行 `alembic upgrade head` 验证 migration 可执行
```

---

## AGENTS.md 完整模板

以下是一个适用于 Python Web 项目的完整模板，涵盖了 Harness Engineering 所需的所有关键信息：

````markdown
# AGENTS.md — Project Agent Constitution
_Last updated: 2026-03-01 | Applies to: All AI Agents working on this codebase_

---

## 🗺️ 项目概览

**项目名称**：[项目名]  
**核心功能**：[一句话描述]  
**技术栈**：Python 3.11 / FastAPI / PostgreSQL / Redis / Docker  
**文档索引**：
- 架构文档：[docs/architecture.md](./docs/architecture.md)
- API 规范：[docs/api-spec.md](./docs/api-spec.md)
- 数据库 Schema：[docs/schema.md](./docs/schema.md)

---

## 🏗️ 架构约束（不可违反）

### 模块依赖规则
```
models/ (数据模型)
  ↓ 只能被
services/ (业务逻辑)
  ↓ 只能被
api/ (路由层)
  ↓ 只能被
main.py
```
**禁止**：api/ 直接调用 models/，services/ 调用 api/。

### 禁止操作清单
- ❌ 禁止修改数据库模型而不创建对应的 Alembic migration
- ❌ 禁止在 api/ 层直接执行 SQL 查询
- ❌ 禁止硬编码任何凭证、API Key 或配置值（使用环境变量）
- ❌ 禁止修改 requirements.txt 而不更新 pyproject.toml
- ❌ 禁止跳过测试或注释掉现有测试来通过 CI

---

## 🧪 测试规范（必须执行）

### 完成任何代码修改后，必须运行：
```bash
# 运行完整测试套件
pytest tests/ -v

# 运行 Lint 检查
ruff check src/
mypy src/

# 类型检查
pyright src/
```

### 测试结构
```
tests/
├── unit/          # 单元测试（不需要数据库）
├── integration/   # 集成测试（需要测试数据库）
└── e2e/           # 端到端测试（需要完整环境）
```

**原则**：修改哪个模块，就运行该模块对应的测试。  
**强制**：不得删除、注释或修改现有测试用例（除非明确修复测试 bug）。

---

## 📦 依赖管理

```bash
# 添加依赖（必须用 Poetry）
poetry add <package>

# 添加开发依赖
poetry add --group dev <package>

# 更新依赖锁文件
poetry lock
```

**禁止**：直接编辑 requirements.txt 或 setup.py。

---

## 🗄️ 数据库操作规范

### Schema 变更流程
```bash
# 1. 修改 models/ 中的 SQLAlchemy 模型
# 2. 生成 migration 文件
alembic revision --autogenerate -m "描述变更"

# 3. 检查生成的 migration 文件（必须人工核查）
# 4. 应用 migration
alembic upgrade head
```

### 查询规范
- 使用 SQLAlchemy ORM，避免原始 SQL
- 复杂查询必须添加适当的索引
- 批量操作使用 `bulk_insert_mappings` 或 `executemany`

---

## 🎨 代码风格

### 命名规范
| 场景 | 规范 | 示例 |
|------|------|------|
| 变量/函数 | snake_case | `user_profile`, `get_user_by_id` |
| 类名 | PascalCase | `UserService`, `OrderRepository` |
| 常量 | SCREAMING_SNAKE | `MAX_RETRY_COUNT` |
| 私有方法 | 下划线前缀 | `_validate_input` |

### 类型提示
```python
# ✅ 所有公共函数必须有完整类型提示
async def create_user(
    user_data: UserCreateSchema,
    db: AsyncSession,
) -> UserSchema:
    ...

# ❌ 不允许
async def create_user(user_data, db):
    ...
```

---

## ⚠️ 已知风险区域

以下区域历史上容易出现问题，修改时需要特别小心：

- `src/services/payment_service.py`：支付逻辑，任何修改必须运行完整集成测试
- `src/models/user.py`：用户模型，修改前检查所有关联 migration
- `migrations/`：不要手动编辑已应用的 migration 文件

---

## 🚨 出现错误时

1. **Lint 错误**：运行 `ruff check src/ --fix` 自动修复大部分问题
2. **类型错误**：检查 `mypy.ini` 中的配置，有些第三方库需要特殊处理
3. **Migration 冲突**：运行 `alembic history` 查看历史，解决分支冲突
4. **测试数据库问题**：运行 `make reset-test-db` 重置测试数据库

---

_此文件由团队维护。如果发现内容过时或错误，请更新此文件并通知相关人员。_
````

---

## 进阶技巧：分层文档结构

对于大型项目，单一的 `AGENTS.md` 会变得难以维护。推荐采用**分层文档**结构：

```
项目根目录/
├── AGENTS.md                    # 入口文件（只有目录和核心规则）
├── docs/
│   ├── architecture.md          # 架构决策记录（ADR）
│   ├── api-conventions.md       # API 规范详解
│   ├── testing-guide.md         # 测试策略详解
│   └── module-guides/
│       ├── payment.md           # 支付模块专项指南
│       └── user-auth.md         # 用户认证专项指南
└── src/
    └── [各模块目录]/
        └── AGENTS.md            # 模块级 Agent 宪法
```

模块级 `AGENTS.md` 示例（`src/payment/AGENTS.md`）：

```markdown
# Payment Module — Agent Guide

## 重要背景
支付模块集成了三个支付渠道：微信支付 V3、支付宝 V2、Stripe。

## 修改此模块时必须知道
1. 任何金额计算使用 Decimal，绝对不用 float
2. 所有金额单位为"分"（整数），在展示层才转换为"元"
3. 所有支付操作必须有幂等键（idempotency_key）
4. 修改后必须运行：`pytest tests/integration/test_payment.py -v`

## 禁止操作
- 禁止在未经测试的情况下修改金额计算逻辑
- 禁止直接修改 payment_records 表的已完成记录

## 支付渠道联调
见 [docs/payment-integration-guide.md](../../docs/payment-integration-guide.md)
```

---

## 常见错误与最佳实践对比

### 错误 1：内容过于宽泛

```markdown
# ❌ 没有操作指导意义
请在编写代码时注意代码质量和可维护性。

# ✅ 具体可执行
完成每次代码修改后，运行以下命令：
1. `pytest tests/ -v --tb=short`（测试）
2. `ruff check src/ --fix`（Lint 修复）
3. `mypy src/ --ignore-missing-imports`（类型检查）
所有命令必须通过，才视为修改完成。
```

### 错误 2：只有规则，没有原因

```markdown
# ❌ 无上下文的禁令
- 禁止修改 payment_service.py 中的 calculate_amount 函数

# ✅ 有原因的规则（Agent 能更好地理解边界）
- ⚠️ payment_service.py 的 calculate_amount 函数包含复杂的多渠道折扣逻辑。
  历史上曾因错误修改导致生产事故。修改此函数前必须：
  1. 阅读 docs/payment-discount-spec.md 中的完整规格说明
  2. 运行 pytest tests/integration/test_payment.py -v 确保全部通过
  3. 提交前获得 @payment-team 的代码审查
```

### 错误 3：文档与代码脱节

`AGENTS.md` 必须与代码库保持同步，否则它不只是"没用"，而且会**积极地误导** Agent。

推荐在 CI 中加入文档一致性检查：

```yaml
# .github/workflows/agents-md-check.yml
name: AGENTS.md Consistency Check

on: [push, pull_request]

jobs:
  check-agents-md:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Check tool references exist
        run: |
          # 提取 AGENTS.md 中引用的所有命令
          grep -oP '`[^`]+`' AGENTS.md | grep -v '^\.' | while read cmd; do
            # 检查命令是否在 Makefile 或 pyproject.toml 中有定义
            echo "Checking: $cmd"
          done
      
      - name: Check doc references exist
        run: |
          # 提取 AGENTS.md 中的所有文件引用，检查文件是否存在
          grep -oP '\[.*?\]\(\.\/.*?\)' AGENTS.md | grep -oP '\(\.\/.*?\)' | \
          tr -d '()' | while read filepath; do
            if [ ! -f "$filepath" ] && [ ! -d "$filepath" ]; then
              echo "❌ AGENTS.md 引用了不存在的文件: $filepath"
              exit 1
            fi
          done
          echo "✅ 所有文件引用有效"
```

---

## AGENTS.md vs CLAUDE.md：如何选择？

| 特性 | AGENTS.md | CLAUDE.md |
|------|-----------|-----------|
| 适用范围 | 所有 AI Agent（通用） | Claude Code 专用 |
| 自动读取 | 需要 Agent 框架配置 | Claude Code 自动读取 |
| 语法支持 | 标准 Markdown | 可使用 Claude 特定标签 |
| 推荐场景 | 多 Agent 系统、跨工具协作 | Claude Code 重度用户 |

**建议**：两个都创建。`CLAUDE.md` 可以是 `AGENTS.md` 的超集，额外包含 Claude Code 专属配置。

```markdown
# CLAUDE.md
<!-- 引用通用宪法 -->
<!-- @include AGENTS.md -->

## Claude Code 专属配置

### 工具使用偏好
- 搜索代码：优先使用 `grep -r` 而非 `find`
- 编辑文件：修改前先用 `cat` 确认内容
- 运行命令：对于长时间运行的命令，添加超时参数

### 自动 Compact 策略
- 当上下文利用率超过 60% 时，主动请求压缩历史
```

---

## 本节小结

| 关键点 | 要点 |
|--------|------|
| **核心目的** | 让 Agent 在任何时候都能获取到正确的项目约定，无需依赖"记忆" |
| **三大原则** | 机器可读、渐进式披露、规范行为而非描述状态 |
| **必包含内容** | 架构约束、禁止操作清单、测试规范、验证命令 |
| **避免** | 过长（>500 行）、纯描述性文字、脱离代码的文档 |
| **维护** | 每次架构决策变更后，同步更新 AGENTS.md |

> 💡 **检验标准**：一个好的 `AGENTS.md` 应该让任何第一次接触这个项目的 AI Agent（或人类工程师），在阅读 5 分钟后就能正确地完成第一个代码修改任务——包括知道要运行哪些测试、不能碰哪些文件。

---

*下一节：[9.4 生产级案例：OpenAI、LangChain、Stripe](./04_production_cases.md)*
