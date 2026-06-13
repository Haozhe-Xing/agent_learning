# 9.3 AGENTS.md / CLAUDE.md: Agent Constitution Writing Guide

> 📜 *"The code repository is the single source of truth. All knowledge guiding AI Agent behavior must be stored in the codebase in machine-readable form."*  
> — OpenAI Codex Team Engineering Practice

---

## What Is AGENTS.md / CLAUDE.md?

In Harness Engineering, **`AGENTS.md`** (or `CLAUDE.md`) is a special file placed in the project root directory, known as the **Agent Constitution**.

Its purpose is: **before the Agent starts working, tell it the rules, architecture, conventions, and boundaries of this project.**

```
Project root/
├── AGENTS.md          ← Agent Constitution (general, applies to all AI Agents)
├── CLAUDE.md          ← Claude-specific version (automatically read by Claude Code)
├── src/
├── tests/
└── ...
```

> **Why is it needed?**  
> Every time an Agent starts a new task, it has "amnesia" — it doesn't know the project's architectural decisions, code style, forbidden operations...  
> If you don't tell it, it will infer from "general best practices," which often doesn't match your project's conventions.

---

## Three Core Principles of the Agent Constitution

Before starting to write content, understand three core principles:

### Principle 1: Machine-Readable, Not Human-Readable

```markdown
# ❌ Bad writing (human-oriented narrative text)
This project uses Python. When writing code, please pay attention to code quality,
maintain good naming habits, and try to write clear comments so that other colleagues
can read it more easily...

# ✅ Good writing (structured, machine-readable)
## Language & Framework
- Language: Python 3.11+
- Web framework: FastAPI
- ORM: SQLAlchemy 2.0
- Testing: pytest + pytest-asyncio

## Code Standards
- Naming: snake_case for variables/functions, PascalCase for class names
- Type hints: all public functions must have complete type hints
- Docstrings: use Google style
```

### Principle 2: Progressive Disclosure, Not All-Inclusive

The OpenAI team found that a 5,000-line `AGENTS.md` actually performs worse than a 200-line `AGENTS.md`. The reason is that Agents get "lost" in large amounts of information.

**Correct approach**: `AGENTS.md` is the "table of contents," with detailed content in separate files:

```markdown
## Architecture Overview
See docs/architecture.md（见 `./docs/architecture.md`）

## API Conventions
See docs/api-conventions.md（见 `./docs/api-conventions.md`）

## Testing Guide  
See docs/testing-guide.md（见 `./docs/testing-guide.md`）
```

### Principle 3: Prescribe Behavior, Not Describe State

```markdown
# ❌ Descriptive (tells the Agent what the current state is)
We use a PostgreSQL database.

# ✅ Behavioral (tells the Agent what to do)
When modifying the database schema:
1. Must create an Alembic migration file; do not modify models directly
2. Migration file naming format: {timestamp}_{description}.py
3. Run `alembic upgrade head` before committing to verify the migration is executable
```

---

## Complete AGENTS.md Template

The following is a complete template for a Python web project, covering all key information required by Harness Engineering:

````markdown
# AGENTS.md — Project Agent Constitution
_Last updated: 2026-03-01 | Applies to: All AI Agents working on this codebase_

---

## 🗺️ Project Overview

**Project Name**: [Project Name]  
**Core Function**: [One-sentence description]  
**Tech Stack**: Python 3.11 / FastAPI / PostgreSQL / Redis / Docker  
**Documentation Index**:
- Architecture docs: docs/architecture.md（见 `./docs/architecture.md`）
- API spec: docs/api-spec.md（见 `./docs/api-spec.md`）
- Database schema: docs/schema.md（见 `./docs/schema.md`）

---

## 🏗️ Architectural Constraints (Non-Negotiable)

### Module Dependency Rules
```
models/ (data models)
  ↓ can only be imported by
services/ (business logic)
  ↓ can only be imported by
api/ (routing layer)
  ↓ can only be imported by
main.py
```
**Forbidden**: api/ directly calling models/, services/ calling api/.

### Forbidden Operations Checklist
- ❌ Forbidden: modifying database models without creating a corresponding Alembic migration
- ❌ Forbidden: executing SQL queries directly in the api/ layer
- ❌ Forbidden: hardcoding any credentials, API keys, or configuration values (use environment variables)
- ❌ Forbidden: modifying requirements.txt without updating pyproject.toml
- ❌ Forbidden: skipping tests or commenting out existing tests to pass CI

---

## 🧪 Testing Standards (Must Execute)

### After any code modification, you must run:
```bash
# Run the complete test suite
pytest tests/ -v

# Run lint checks
ruff check src/
mypy src/

# Type checking
pyright src/
```

### Test Structure
```
tests/
├── unit/          # Unit tests (no database needed)
├── integration/   # Integration tests (requires test database)
└── e2e/           # End-to-end tests (requires full environment)
```

**Principle**: when modifying a module, run the tests corresponding to that module.  
**Mandatory**: do not delete, comment out, or modify existing test cases (unless explicitly fixing a test bug).

---

## 📦 Dependency Management

```bash
# Add dependency (must use Poetry)
poetry add <package>

# Add dev dependency
poetry add --group dev <package>

# Update dependency lock file
poetry lock
```

**Forbidden**: directly editing requirements.txt or setup.py.

---

## 🗄️ Database Operation Standards

### Schema Change Process
```bash
# 1. Modify the SQLAlchemy model in models/
# 2. Generate migration file
alembic revision --autogenerate -m "describe the change"

# 3. Review the generated migration file (must be manually checked)
# 4. Apply migration
alembic upgrade head
```

### Query Standards
- Use SQLAlchemy ORM, avoid raw SQL
- Complex queries must add appropriate indexes
- Batch operations use `bulk_insert_mappings` or `executemany`

---

## 🎨 Code Style

### Naming Conventions
| Context | Convention | Example |
|---------|-----------|---------|
| Variables/functions | snake_case | `user_profile`, `get_user_by_id` |
| Class names | PascalCase | `UserService`, `OrderRepository` |
| Constants | SCREAMING_SNAKE | `MAX_RETRY_COUNT` |
| Private methods | underscore prefix | `_validate_input` |

### Type Hints
```python
# ✅ All public functions must have complete type hints
async def create_user(
    user_data: UserCreateSchema,
    db: AsyncSession,
) -> UserSchema:
    ...

# ❌ Not allowed
async def create_user(user_data, db):
    ...
```

---

## ⚠️ Known Risk Areas

The following areas have historically been prone to issues — exercise extra caution when modifying:

- `src/services/payment_service.py`: payment logic; any modification must run the complete integration tests
- `src/models/user.py`: user model; check all associated migrations before modifying
- `migrations/`: do not manually edit already-applied migration files

---

## 🚨 When Errors Occur

1. **Lint errors**: run `ruff check src/ --fix` to auto-fix most issues
2. **Type errors**: check the configuration in `mypy.ini`; some third-party libraries need special handling
3. **Migration conflicts**: run `alembic history` to view history and resolve branch conflicts
4. **Test database issues**: run `make reset-test-db` to reset the test database

---

_This file is maintained by the team. If you find outdated or incorrect content, please update this file and notify relevant personnel._
````

---

## Advanced Technique: Layered Documentation Structure

For large projects, a single `AGENTS.md` can become difficult to maintain. A **layered documentation** structure is recommended:

```
Project root/
├── AGENTS.md                    # Entry file (only table of contents and core rules)
├── docs/
│   ├── architecture.md          # Architecture Decision Records (ADR)
│   ├── api-conventions.md       # API specification details
│   ├── testing-guide.md         # Testing strategy details
│   └── module-guides/
│       ├── payment.md           # Payment module specific guide
│       └── user-auth.md         # User authentication specific guide
└── src/
    └── [each module directory]/
        └── AGENTS.md            # Module-level Agent Constitution
```

Module-level `AGENTS.md` example (`src/payment/AGENTS.md`):

```markdown
# Payment Module — Agent Guide

## Important Background
The payment module integrates three payment channels: WeChat Pay V3, Alipay V2, and Stripe.

## Must Know When Modifying This Module
1. Use Decimal for any amount calculations; never use float
2. All amounts are in "cents" (integers); convert to "dollars" only in the display layer
3. All payment operations must have an idempotency_key
4. After modification, must run: `pytest tests/integration/test_payment.py -v`

## Forbidden Operations
- Forbidden: modifying amount calculation logic without testing
- Forbidden: directly modifying completed records in the payment_records table

## Payment Channel Integration Testing
See `docs/payment-integration-guide.md`
```

---

## Common Mistakes vs. Best Practices

### Mistake 1: Content Too Vague

```markdown
# ❌ No actionable guidance
Please pay attention to code quality and maintainability when writing code.

# ✅ Specific and executable
After each code modification, run the following commands:
1. `pytest tests/ -v --tb=short` (tests)
2. `ruff check src/ --fix` (lint fix)
3. `mypy src/ --ignore-missing-imports` (type check)
All commands must pass before the modification is considered complete.
```

### Mistake 2: Rules Without Reasons

```markdown
# ❌ Prohibition without context
- Forbidden: modifying the calculate_amount function in payment_service.py

# ✅ Rules with reasons (Agent can better understand boundaries)
- ⚠️ The calculate_amount function in payment_service.py contains complex multi-channel discount logic.
  Historically, incorrect modifications have caused production incidents. Before modifying this function:
  1. Read the complete specification in docs/payment-discount-spec.md
  2. Run pytest tests/integration/test_payment.py -v to ensure all tests pass
  3. Get a code review from @payment-team before committing
```

### Mistake 3: Documentation Disconnected from Code

`AGENTS.md` must stay in sync with the codebase; otherwise it's not just "useless" — it will **actively mislead** the Agent.

It's recommended to add documentation consistency checks to CI:

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
          # Extract all commands referenced in AGENTS.md
          grep -oP '`[^`]+`' AGENTS.md | grep -v '^\.' | while read cmd; do
            # Check if the command is defined in Makefile or pyproject.toml
            echo "Checking: $cmd"
          done
      
      - name: Check doc references exist
        run: |
          # Extract all file references in AGENTS.md, check if files exist
          grep -oP '\[.*?\]\(\.\/.*?\)' AGENTS.md | grep -oP '\(\.\/.*?\)' | \
          tr -d '()' | while read filepath; do
            if [ ! -f "$filepath" ] && [ ! -d "$filepath" ]; then
              echo "❌ AGENTS.md references a non-existent file: $filepath"
              exit 1
            fi
          done
          echo "✅ All file references are valid"
```

---

## AGENTS.md vs CLAUDE.md: How to Choose?

| Feature | AGENTS.md | CLAUDE.md |
|---------|-----------|-----------|
| Scope | All AI Agents (general) | Claude Code specific |
| Auto-read | Requires Agent framework configuration | Claude Code reads automatically |
| Syntax support | Standard Markdown | Can use Claude-specific tags |
| Recommended for | Multi-Agent systems, cross-tool collaboration | Heavy Claude Code users |

**Recommendation**: create both. `CLAUDE.md` can be a superset of `AGENTS.md`, additionally including Claude Code-specific configurations.

```markdown
# CLAUDE.md
<!-- Reference the general constitution -->
<!-- @include AGENTS.md -->

## Claude Code Specific Configuration

### Tool Usage Preferences
- Searching code: prefer `grep -r` over `find`
- Editing files: use `cat` to confirm content before modifying
- Running commands: add timeout parameters for long-running commands

### Auto Compact Strategy
- When context utilization exceeds 60%, proactively request history compression
```

---

## Summary

| Key Point | Summary |
|-----------|---------|
| **Core purpose** | Ensure the Agent can always access the correct project conventions, without relying on "memory" |
| **Three principles** | Machine-readable, progressive disclosure, prescribe behavior not describe state |
| **Must include** | Architectural constraints, forbidden operations checklist, testing standards, validation commands |
| **Avoid** | Too long (>500 lines), purely descriptive text, documentation disconnected from code |
| **Maintenance** | Update AGENTS.md synchronously after every architectural decision change |

> 💡 **Quality standard**: a good `AGENTS.md` should enable any AI Agent (or human engineer) encountering this project for the first time to correctly complete their first code modification task after 5 minutes of reading — including knowing which tests to run and which files not to touch.

---

*Next: [9.4 Production Case Studies: OpenAI, LangChain, Stripe](./04_production_cases.md)*
