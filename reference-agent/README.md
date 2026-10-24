# reference-agent

《从零开始学 Agent 开发》第 3、10–11、17–23 章可复用的**最小参考底座**。

本目录实现最小工具循环、固定任务评估、基础防护、MCP 服务和 HTTP 服务。它不依赖 LangChain 或 LangGraph，也不包含记忆、RAG 或三个领域项目的完整业务实现。章节中的框架示例与业务脚手架需按各自说明组装；配套入口见 `examples/README.md`。

## 设计原则（诚实化 + 可运行化）

1. **真实文件存在**：代码在 `src/`，不在 Markdown 中。
2. **默认离线可跑**：无 API Key 时使用 `FakeProvider`，逻辑可测；设 `AGENT_REAL=1` + `OPENAI_API_KEY` 才接真实模型。
3. **一条命令跑测试**：`pytest` 验证随附的固定离线行为；真实模型效果需另行评估。
4. **有界词句筛查**：类型/编码/长度无效或规则命中时阻断；没有命中不代表语义安全，业务授权与资源权限须另外执行。
5. **时效事实可核验**：版本、价格、安装量等说法以官方来源为准，不引用无法独立核验的数字。

## 目录结构

```
reference-agent/
├── pyproject.toml
├── src/reference_agent/
│   ├── provider.py        # LLM 抽象：FakeProvider（离线）/ OpenAIProvider（真实）
│   ├── tools/             # Tool 基类 + CalculatorTool + SearchTool
│   ├── agent.py           # 最小 ReAct 循环，真实执行工具
│   ├── security/          # InjectionGuard（词句筛查）
│   ├── service/           # FastAPI 服务：/chat 真返回、/stream SSE 分块传输
│   ├── mcp_server.py      # 最小可运行 MCP Server（stdio + JSON-RPC）
│   ├── evaluation/        # 固定任务集 + 通过率报告
│   └── data/eval_samples.jsonl  # 随包分发的固定任务集
├── tests/                 # 覆盖 agent/security/evaluation/service/mcp
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
| 3 工具调用、10–11 编排原理对照 | `agent.py` / `tools/`（独立实现，非框架示例） |
| 17 协议（MCP） | `mcp_server.py` |
| 19 安全 | `security/guardrails.py` |
| 20 部署 | `service/app.py` + `docker/Dockerfile` |
| 18 评估 | `evaluation/harness.py` |
| 21–23 综合项目 | 以本底座为共享基础，按任务域扩展 |

## 工具许可与失败验收

`Agent(provider, tools=None)` 使用默认工具；显式传入 `tools=[]` 表示不提供任何工具。执行器也用这份注册表查找工具，因此模型仍发出计算器或搜索请求时，只会收到 `unknown tool`，不会恢复默认权限。`max_iterations` 限制模型调用轮数；用尽后返回任务未完成的说明。

MCP 示例先校验请求、ID、工具名和参数对象；非法输入返回明确错误，不能让同一 stdio 服务退出。它只实现本地教学所需的方法与固定协议声明，不代表完整 MCP 兼容性。书稿第17.5节给出同进程“坏请求 → 正常请求”的运行方法。

在仓库根目录运行对应反例：

```bash
PYTHONPATH=reference-agent/src python -m pytest reference-agent/tests/test_agent.py reference-agent/tests/test_mcp_server.py -q
```

贯穿项目和开发团队的完成状态、独立验收及预算规则见 [数据与完成契约](../examples/CONTRACTS.md)。工具执行受限也不保证最终回答正确；任务是否达标需要独立验收。

## 已知边界（不夸大）

- `FakeProvider` 只覆盖演示所需的确定性行为，不是真实模型。
- `/stream` 在完整答案生成后按空白切分发送 SSE，用于演示传输格式，不是模型 token 的实时流式生成。
- 安全守卫是**起点**，不是生产级安全方案；上线前需结合威胁建模、日志、人工审核。
- 综合项目（coding / data / multimodal）的"组件"在书中以代码片段给出，需读者自行组装为模块；本目录提供的是可运行的最小底座。

## 真实世界项目参考：OpenHarness

`reference-agent` 是**教学用最小底座**，刻意保持精简。想研究一个**真实、生产级、开源**的 Agent Harness 如何组织代码，推荐阅读 **OpenHarness**（HKUDS 出品，MIT 协议）：

- 仓库：<https://github.com/HKUDS/OpenHarness>
- 中文文档：<https://github.com/HKUDS/OpenHarness/blob/main/README.zh-CN.md>
- 它实现了工具调用、Agent Loop、权限治理、生命周期钩子、多智能体协调、持久记忆与上下文压缩，是本书第 8 章 Harness 工程的"可阅读源码"案例。

> 本书第 8 章 8.4 节以 OpenHarness 为例，逐模块对照了六大工程支柱与真实代码目录的映射。
