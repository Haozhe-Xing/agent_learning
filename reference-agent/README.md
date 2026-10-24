# reference-agent

《从零开始学 Agent 开发》第 12–23 章的**统一、可运行**参考实现。

本书前半部分（概念、RAG、规划、记忆等）重在讲清原理；后半部分（框架实战、多 Agent、协议、评估、安全、部署、综合项目）的每个"实战"小节都以本目录的代码为**唯一可信来源**——不再把未验证的伪代码嵌在书里。

## 设计原则（诚实化 + 可运行化）

1. **真实文件存在**：代码在 `src/`，不在 Markdown 中。
2. **默认离线可跑**：无 API Key 时使用 `FakeProvider`，逻辑可测；设 `AGENT_REAL=1` + `OPENAI_API_KEY` 才接真实模型。
3. **一条命令跑测试**：`pytest` 全绿即代表可运行。
4. **安全默认 fail-closed**：注入守卫解析/判定不确定时阻断，而非放行。
5. **时效事实可核验**：版本、价格、安装量等说法以官方来源为准，不引用无法独立核验的数字。

## 目录结构

```
reference-agent/
├── pyproject.toml
├── src/reference_agent/
│   ├── provider.py        # LLM 抽象：FakeProvider（离线）/ OpenAIProvider（真实）
│   ├── tools/             # Tool 基类 + CalculatorTool + SearchTool
│   ├── agent.py           # 最小 ReAct 循环，真实执行工具
│   ├── security/          # InjectionGuard（fail-closed）
│   ├── service/           # FastAPI 服务：/chat 真返回、/stream 真流式
│   ├── mcp_server.py      # 最小可运行 MCP Server（stdio + JSON-RPC）
│   └── evaluation/        # 固定任务集 + 通过率报告
├── tests/                 # 16 个测试，覆盖 agent/security/evaluation/service/mcp
├── data/eval_samples.jsonl
└── docker/Dockerfile
```

## 快速开始

```bash
cd reference-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 跑测试（默认离线，无需 API Key）
pytest -q

# 起真实服务（离线桩）
uvicorn reference_agent.service.app:app --port 8000

# 接真实模型
AGENT_REAL=1 OPENAI_API_KEY=sk-... uvicorn reference_agent.service.app:app --port 8000
```

## 哪些章节用它

| 章节 | 对应模块 |
|------|---------|
| 12–13 框架实战 | `agent.py` / `tools/` |
| 17 协议（MCP） | `mcp_server.py` |
| 19 安全 | `security/guardrails.py` |
| 20 部署 | `service/app.py` + `docker/Dockerfile` |
| 18 评估 | `evaluation/harness.py` |
| 21–23 综合项目 | 以本底座为共享基础，按任务域扩展 |

## 已知边界（不夸大）

- `FakeProvider` 只覆盖演示所需的确定性行为，不是真实模型。
- 安全守卫是**起点**，不是生产级安全方案；上线前需结合威胁建模、日志、人工审核。
- 综合项目（coding / data / multimodal）的"组件"在书中以代码片段给出，需读者自行组装为模块；本目录提供的是可运行的最小底座。

## 真实世界项目参考：OpenHarness

`reference-agent` 是**教学用最小底座**，刻意保持精简。想研究一个**真实、生产级、开源**的 Agent Harness 如何组织代码，推荐阅读 **OpenHarness**（HKUDS 出品，MIT 协议）：

- 仓库：<https://github.com/HKUDS/OpenHarness>
- 中文文档：<https://github.com/HKUDS/OpenHarness/blob/main/README.zh-CN.md>
- 它实现了工具调用、Agent Loop、权限治理、生命周期钩子、多智能体协调、持久记忆与上下文压缩，是本书第 8 章 Harness 工程的"可阅读源码"案例。

> 本书第 8 章 8.4 节以 OpenHarness 为例，逐模块对照了六大工程支柱与真实代码目录的映射。
