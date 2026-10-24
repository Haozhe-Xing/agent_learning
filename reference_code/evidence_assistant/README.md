# 本地资料调研助手

这是本书的独立参考项目。实现、样例资料、测试和数据契约都在本目录，可把整个目录复制出来学习和运行。使用 Python 3.11+，默认仅用标准库，不需要 API Key。

## 要完成的任务

你拿到虚构的“青禾项目”资料，需要回答：上线验收需要什么、调用预算和超时怎样限制、部署失败如何回退。助手按问题建立计划，从多份本地 Markdown/TXT 中检索证据，控制送入回答器的上下文大小，生成带来源的报告。

默认回答器直接摘录实际资料，用于观察数据流与失败处理。报告分别检查“摘录有原文支撑”和“回答覆盖任务要求”。默认样例另有冻结的15个必答要点；改用自己的资料时，也要另行提供任务规格或独立验证器。缺少验收要求时，找到有效摘录仍保留为未完成。

## 先运行一个完整任务

在本目录运行：

```bash
python run.py
python run.py --format json
python run_tests.py
```

也可以从仓库根目录运行：

```bash
python -m reference_code.evidence_assistant
python -m reference_code.evidence_assistant --format json --output /tmp/evidence-report.json
python reference_code/evidence_assistant/run_tests.py
```

默认资料位于 [sample_data/](sample_data/)，与当前工作目录无关。CLI识别该目录时，自动使用 [sample_task_specs.json](sample_task_specs.json) 的3道题、15个冻结要点。规格只供验收，不进入检索资料或回答上下文。输出文件只在显式传入 `--output` 时写入。

逐步操作与结果解读见 [WALKTHROUGH.md](WALKTHROUGH.md)。输入、任务规格和状态字段见 [CONTRACTS.md](CONTRACTS.md)。

## 难度与学习顺序

主路径是一个单进程 CLI、几份本地资料和一份报告。需要用到 Python 函数、字典、文件读取和简单类；逐步练习模块分工、状态传递、预算与验收。

| 步骤 | 实际练习 | 完成标准 |
|---|---|---|
| 1. 运行与改资料 | 跑默认任务，删除一条必答内容后再运行 | 摘录随资料变化；冻结规格应报告缺少的要点，不随回答自动改预期 |
| 2. 改问题与检索 | 用重复的 `--question` 提供多个问题，独立配置任务要求 | 核对命中分数与匹配词；无证据或覆盖未知时整体不宣布完成 |
| 3. 理解执行与预算 | 降低 `--max-steps`、`--max-calls` 或 `--context-bytes` | 保留失败/未完成原因，退出码为非零 |
| 4. 增加一个小能力 | 调整检索排序或增加一种报告字段，并补充相应验收用例 | 原有成功和失败测试继续通过 |
| 选做 | 接入自己的模型回答器与独立任务验证器，或学习 SQLite 偏好记忆 | 单独验证模型质量、记忆隔离和失败行为 |

先按 `research.py → rag.py → planning.py → context.py` 阅读主路径，再查看 `task_acceptance.py` 的独立任务要求与 `harness.py` 的执行约束。其余基础模块可以按章节需要回查。框架、Web 界面、多 Agent 和远程部署作为后续扩展；完成本项目不需要先搭建这些组件。

例如，在仓库根目录提出两个有资料依据的问题：

```bash
python -m reference_code.evidence_assistant \
  --docs reference_code/evidence_assistant/sample_data \
  --question "上线验收需要满足哪些条件？" \
  --question "部署失败后如何回退？"
```

自定义资料使用 `--docs DIR --task-spec FILE`，任务JSON的 `question` 必须与命令中的问题对应；除空白和大小写归一化外，不自动匹配同义问法。每个 `requirements` 要点用 `all_of` 列出必须出现的字面片段。应由任务维护者在运行前独立确定要求，具体格式见契约；不要从生成的回答反推预期，也不要把任务规格当成证据。

若自定义资料未传 `--task-spec`，且没有在Python调用中提供独立 `verifier`，即使 `verification.status=passed`，单题仍为 `evidence_found`、`coverage.status=unknown`，整体 `accepted=false`，退出码为 `1`。CLI不提供验证器参数；Python接口可使用 `ResearchAssistant(..., verifier=...)`。

观察预算停止与缺少证据：

```bash
python -m reference_code.evidence_assistant --max-calls 1
python -m reference_code.evidence_assistant --question "火星矿石运输许可证如何办理？"
```

这两条是失败练习，应返回非零退出码。参数详情见 `python run.py --help`。

## 从报告定位问题

`--format json` 的 `findings` 保留逐题检查信息：

| 字段 | 怎样读取 |
|---|---|
| `retrieval.hits` | 来源、`score`与`matched_terms`。分数是不同匹配词项的数量，不是正确率或置信度；中文通用问句短语被过滤，多片段中文查询通常至少需两个双字片段匹配。这是字符片段匹配，不是真正中文分词 |
| `verification` | 逐行核对来源、原文摘录和词项重合；`passed`只表示证据支撑检查通过 |
| `coverage` | 与独立任务要求/验证器比较；`matched_points`、`missing_points`和`details`定位覆盖与缺口；没有匹配规格或验证器则为`unknown` |
| `status`、`stage`、`reason` | `answered`为两项检查均通过；`partial`有有效摘录但任务要求未通过；`evidence_found`表示覆盖未知。失败状态及原因码保留发生阶段 |
| `summary`、`budgets`、`failure`（报告顶层） | 汇总已验收/未完成题数、实际步骤/调用/字节用量；`failure.question_index`从0开始，`step`、`code`和`message`指出首个未完成位置 |

`no_evidence`、`generation_failed`、`verification_failed`与上下文超限分别保留原因。步骤或工具预算停止时，部分题可能还在 `not_started` / `retrieved`，应结合顶层 `failure` 和工具轨迹阅读。只有计划完成且所有题的原文支撑、任务覆盖均通过，整体才为 `completed`、`accepted=true`，CLI退出 `0`。

负责人生日问题可能命中负责人相关段落，却没有生日答案：无本题要求时仍为覆盖`unknown`，不是自动归入`no_evidence`。后者仅表示没有合格检索命中；两种情况都不能宣布任务完成。

冻结要点做空白/大小写归一化后的字面包含检查。它能指出本例规定的缺项，不能普遍识别同义改写、否定关系、矛盾或完整自然语言语义；原文匹配也不证明资料真实。扩展任务时应另设代表性验收集与合适的独立验证器。

## 文件分工

```text
evidence_assistant/
├── README.md / WALKTHROUGH.md / CONTRACTS.md  # 入口、操作与验收约定
├── run.py / __main__.py      # 同一调研应用的启动方式
├── research.py              # 主应用：读取、计划、检索、上下文、报告
├── sample_data/             # 可替换的虚构项目资料
├── sample_task_specs.json   # 默认三题的15个冻结必答要点，不是证据
├── task_acceptance.py       # 独立任务覆盖检查，不参与检索/生成
├── rag.py                   # 词法检索、引用和可替换回答器
├── planning.py              # 有预算的计划执行
├── context.py               # 实际消息的字节预算
├── harness.py               # 工具许可、调用预算与宿主验收
├── memory.py                # SQLite 与分层记忆基础
├── research_agent.py        # 章节用的简化多问题编排
├── agentic_rag.py           # 章节用的受限查询重试
├── demo.py                  # 六阶段距离计算基础练习
├── contracts.py / acceptance.py  # 基础练习的提取、计算与独立验收
├── tests/                   # 行为测试与独立预期资料
└── run_tests.py             # 本目录的标准库测试入口
```

`examples/` 中的章节脚本复用这里的模块。旧 `examples.learning_path` 仅转发到这里，不维护第二份实现。协议与 HTTP 服务的另一个参考底座见仓库 [reference-agent](../../reference-agent/README.md)。

## 更小的基础练习

初学工具、记忆、规划、RAG、上下文或 Harness 时，可以先用单条距离资料观察六次能力增量。从仓库根目录运行：

```bash
python -m reference_code.evidence_assistant.demo --all
python -m reference_code.evidence_assistant.demo --stage memory --output-unit milliseconds
```

独立复制项目后，在 `evidence_assistant/` 的父目录运行基础练习：

```bash
python -m evidence_assistant.demo --all
```

两种任务分别用于完整项目体验与机制学习，不将它们当作同题模型评测。

输入格式、引用的证明范围与停止条件见 [CONTRACTS.md](CONTRACTS.md)。仓库整体的实测记录见 [examples/VALIDATION.md](../../examples/VALIDATION.md)。

## English quick start

Copy this self-contained, standard-library project and run `python run.py`, then `python run_tests.py`. Bundled documents automatically use `sample_task_specs.json`: three questions with 15 frozen requirements. Custom documents need `--task-spec FILE` or a separately supplied Python verifier. Without either, supported excerpts remain `evidence_found`, coverage is `unknown`, and the CLI exits 1. Retrieval scores count matched lexical terms. Source support (`verification`) and task coverage (`coverage`) are separate checks; only both passing for every question yields `accepted=true`. Literal requirements do not establish general semantic completeness or source truth. See the walkthrough and contracts for input, failure, and budget details.
