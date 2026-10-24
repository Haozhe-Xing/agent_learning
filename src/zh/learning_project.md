# 贯穿项目与实战验收

先运行一个完整的小应用，再拆开理解它。独立参考项目 `reference_code/evidence_assistant/` 的主入口是**本地资料调研助手**：读取几份虚构项目文档，依次回答多个问题，把检索证据、计划状态、上下文预算记录和引用组织为一份报告。

它适合已经会运行 Python、读函数和字典的初学者继续练习。难度来自模块之间真实的数据交接与失败处理；默认使用标准库、离线资料和原文摘录，不要求先配置模型、数据库或框架。它没有实现开放网络研究、事实交叉核查、长期用户记忆、Web 服务或全部框架适配；这些按需要再扩展。

## 先运行，再替换资料和问题

在仓库根目录使用 Python 3.11+：

```bash
python -m reference_code.evidence_assistant
```

默认读取项目内 `sample_data/` 的虚构资料，并自动使用 `sample_task_specs.json` 中3道题的15个冻结要点：上线验收4个、工具预算与超时6个、失败回退5个。资料参与检索和回答；任务规格只供独立验收，不作为证据送入回答器。报告分别说明摘录是否有原文支撑，以及是否覆盖这些要求。

再用两份练习资料、两个问题和另外制定的任务规格替换默认输入。下面用独立临时目录保存本次练习，任务JSON放在证据目录外：

```bash
research_demo_dir=$(mktemp -d "${TMPDIR:-/tmp}/evidence-study-zh.XXXXXX")
mkdir "$research_demo_dir/docs"
cat > "$research_demo_dir/docs/release.md" <<'EOF'
上线验收条件：完成回归测试，并由项目负责人确认回退演练记录。
EOF
cat > "$research_demo_dir/docs/rollback.txt" <<'EOF'
部署失败后应先停止新增流量，再回退到上一稳定版本，并保留失败日志。
EOF
cat > "$research_demo_dir/tasks.json" <<'EOF'
{
  "schema_version": 1,
  "tasks": [
    {
      "question": "上线验收需要满足哪些条件？",
      "requirements": [
        {"id": "regression", "description": "完成回归测试", "all_of": ["完成回归测试"]},
        {"id": "approval", "description": "项目负责人确认回退演练记录", "all_of": ["由项目负责人确认回退演练记录"]}
      ]
    },
    {
      "question": "部署失败后如何回退？",
      "requirements": [
        {"id": "traffic", "description": "停止新增流量", "all_of": ["停止新增流量"]},
        {"id": "restore", "description": "回退到上一稳定版本", "all_of": ["回退到上一稳定版本"]},
        {"id": "logs", "description": "保留失败日志", "all_of": ["保留失败日志"]}
      ]
    }
  ]
}
EOF
python -m reference_code.evidence_assistant \
  --docs "$research_demo_dir/docs" \
  --task-spec "$research_demo_dir/tasks.json" \
  --question '上线验收需要满足哪些条件？' \
  --question '部署失败后如何回退？' --format json
```

该例应退出`0`，两题均为`answered`，原文支撑与覆盖均`passed`，整体`accepted=true`。JSON中的`question`与命令保持一致；除空白和大小写归一化外，不自动匹配同义问法。每个要点的`all_of`列出必须全部出现的字面片段。实际项目应由任务维护者在生成回答前独立制定要求，不能看完答案再凑一份可通过的预期。

去掉`--task-spec`再运行自定义资料，即使摘录完全有出处，结果仍应是`evidence_found`、`coverage.status=unknown`、`accepted=false`，退出`1`。CLI仅对包内默认资料自动选择冻结规格；Python调用方也可显式提供独立`verifier`。

支持目录顶层的1–32份UTF-8 `.md` / `.txt`普通文件，以及重复传入的1–4个`--question`。资料按非空段落形成来源，例如`release.md:p1`；检索用过滤后的中文相邻双字片段或英文词匹配。替换原文中的“上一稳定版本”，摘录应随之变化；冻结要求也会报告字面要点缺失。只有业务要求经过独立确认后才更新规格。

`retrieval.hits`列出每段的`score`和`matched_terms`：分数是不同匹配词项的数量，不是置信度或回答正确率。中文通用问句短语先过滤，多片段中文查询通常需至少两个双字片段匹配，这不是真正中文分词。命中同主题材料仍可能没有答案：询问负责人生日可能命中负责人段落，覆盖保持未知；完全无命中才是`no_evidence`。本例不声称能完美识别所有无答案问题。

`--max-steps` 限制计划步骤，`--max-calls` 限制工具调用，`--context-bytes` 限制实际消息的 UTF-8 字节数。默认分别为 8、8、8192。将默认任务的 `--max-steps` 改为 `1`，应得到未完成结果。还应尝试问题无命中或必需证据放不进上下文的输入，确认失败不能变成成功报告。`--output FILE` 可保存报告，目标文件已存在时拒绝覆盖。

结果将`status`与`accepted`分开记录。逐题检查`findings`中的`verification`（原文支撑）和`coverage`（独立任务覆盖），后者的`matched_points`、`missing_points`、`details`定位遗漏。有有效摘录但缺必答点为`partial`，未配置本题要求为`evidence_found`/覆盖`unknown`；检索、生成或摘录核对失败另有失败状态。`stage`和`reason.code/message`说明发生位置。

顶层`summary`统计已验收和未完成题数，`budgets`记录实际用量，`failure`保留首个未完成问题（`question_index`从0开始）、步骤与原因。步骤/调用预算停止时，某些题还会停在`not_started`或`retrieved`；上下文超限可比较`required_bytes`与`limit_bytes`。结合`execution`和`tool_trace`读取，不能只看执行计划已完成。只有所有问题两层验收均通过，整体才为`completed`、`accepted=true`且退出`0`。

离线回答是受控的原文摘录，冻结要点是字面包含检查；它们不证明资料真实、同义改写正确、否定或矛盾已被理解，也不是通用语义覆盖证明。阅读`verification_scope`，增加业务要求时另设代表性任务与合适的独立验收。

## 用四个模块读懂主路径

先按下面顺序阅读项目目录中的文件；不必先读完全部实现。

| 文件 | 先追踪的问题 | 动手改动 |
|---|---|---|
| `research.py` | 文件和问题怎样进入任务？每个问题怎样产生检索、回答步骤，结果怎样组成报告？ | 换文档、问题或报告格式，确认失败仍保留原因 |
| `rag.py` | 命中文档与回答实际引用的来源怎样区分？空回答、错误引用怎样被拒绝？ | 用无命中、缺引用和未知引用案例观察阶段状态 |
| `planning.py` | 步骤结果怎样传给后续步骤？失败或预算耗尽怎样停止？ | 降低步骤预算，核对轨迹与最终未完成状态 |
| `context.py` | 回答消费的是哪一份消息？必需资料超出预算时怎么办？ | 降低字节预算，确认必需证据不会被静默丢弃 |

随后看`task_acceptance.py`与`sample_task_specs.json`，追踪独立要求怎样决定覆盖；再看`harness.py`的执行约束，并对照`tests/test_research.py`和`tests/test_retrieval_and_tasks.py`中的反例。

```text
reference_code/evidence_assistant/
├── research.py / rag.py / planning.py / context.py
├── task_acceptance.py       # 独立任务覆盖检查
├── sample_task_specs.json   # 默认3题、15个冻结要点，不是证据
├── sample_data/             # 实际供检索与摘录的资料
├── harness.py / tests/
└── README.md / WALKTHROUGH.md / CONTRACTS.md
```

`memory.py`的持久偏好、`demo.py`的六阶段计算，以及`acceptance.py` / `contracts.py`的数值契约是基础机制或进阶阅读，不是第一次运行调研应用的前置要求。

## 基础练习：同一道计算题观察六次迭代

前几章仍用受控的月球距离题逐项观察机制。它与调研助手是两项不同任务：前者便于固定数值预期，后者练习多资料、多问题与报告交付；不要混用两者的业务验收。

```bash
python -m reference_code.evidence_assistant.demo --all
```

结果标记为 `offline-fixture`：读取虚构的 `384400 km` 距离资料，提取数量、归一化单位、计算光行时间并引用来源。默认约为 1.28222 秒；它验证契约和编排，不测真实模型质量。

| 迭代 | 对应章节 | 增加的产物与失败检查 |
|---|---|---|
| tools | 第3章 | 原文数量、单位与计算；资料变化必须改变结果 |
| memory | 第4章 | 保存并读回的输出单位；检查重开、隔离和删除 |
| planning | 第5章 | 步骤依赖、结果与状态；失败和超限不能完成 |
| rag | 第6章 | 命中原文与回答引用；无证据时报告缺失 |
| context | 第7章 | 实际消息、字节预算与取舍；必需证据不能静默丢弃 |
| harness | 第8章 | 工具轨迹与独立验收；未授权调用和错误产物被拒绝 |

用 `--stage memory` 可只看一次迭代。也可验证资料确实参与计算：

```bash
printf 'Moon distance is 100 km.\n' > /tmp/light-distance.txt
python -m reference_code.evidence_assistant.demo --all --file /tmp/light-distance.txt --output-unit milliseconds
```

六阶段均应输出约 `0.33356409519815205` 毫秒；把 `100 km` 改成 `100000 m` 后结果相同。该适配器只支持受控十进制距离语句，最多 24 位数字，归一化范围为 `[1e-12, 1e18]` 米；完整语法见项目 `CONTRACTS.md`。

计算任务的独立验收器重新解析原始资料，不调用生产端解析器或计算器，并核对数值、单位、计算、回答、来源和原文摘要。固定预期位于 `tests/fixtures/light_travel_cases.json`，运行路径不读取它。摘要保证输入一致性，不能证明资料真实；QA 默认引用检查也不能代替语义支持验证。

## 两个关键过渡

**记忆 → 规划（Memory to planning）**：在基础计算题中，记住“输出毫秒”不能阻止漏做数量提取、重复计算或提前结束。规划增加步骤依赖、结果状态和停止规则。故意把步骤预算设得过低，确认系统报告未完成。调研应用复用同一个原则，为每个问题安排检索与回答；它没有因此自动获得跨会话记忆。

**上下文 → Harness（Context to Harness）**：在同一道计算题中，即使必需原文放进消息且满足大小预算，也不能据此授权额外工具或宣布错误数值正确。Harness 在宿主执行前后检查许可、调用预算与产物。调研应用同样需要这些执行约束，同时按自己的原文证据契约验收，不能套用光行时间的数值判据。

## 全书的交接路线

| 阶段 | 接过什么、交出什么 | 已有实现与读者工作 |
|---|---|---|
| [1–2 基础](./part1.md) | 从需求形成任务卡、调用基线和失败记录 | 环境入口已提供；实际调用记录由读者保存 |
| [3–8 核心迭代](./part2.md) | 用基础题观察机制，再在调研应用追踪检索、计划、上下文与报告 | 上述两条离线路径及反例测试已提供 |
| [9 技能复用](./chapter_skill/README.md) | 把已验收流程整理为不扩权的技能契约 | 设计与练习；没有完整技能服务入口 |
| [10–11 实现选择](./part3.md) | 保留任务、权限和验收，再决定是否迁移实现 | 已有框架示例使用其他任务；调研应用的同题适配由读者完成 |
| [12–15 产品与架构](./part3.md) | 根据具体需求写采用、跳过及机制迁移记录 | 案例选读；没有四产品统一适配器 |
| [16–17 协作与集成](./part4.md) | 从单 Agent 基线走向必要的分工或跨系统接口 | 开发团队与最小 stdio 协议可运行；同题收益另测 |
| [18–20 验证与部署](./part5.md) | 将轨迹、失败和验收变成门禁与运行边界 | 另有 `reference-agent/` 评估和 HTTP/SSE 底座；本应用的服务、鉴权、负载与恢复仍是部署作业 |
| [21–23 领域交付](./part6.md) | 选择一个领域并交付独立验收报告 | 设计与组装分支；需要实现相应实际组件 |
| [24–25 研究进阶](./part7.md) | 从测得的瓶颈与冻结基线出发做训练或候选优化 | 在应用交付后按需研究，不是前置课程 |

首次开发可沿 **1–8 → 需要时10或11 → 18–20 → 任选21–23** 前进。第9章与12–17章按复用、产品、协作和集成需求选读。使用[问题决策表](./part3.md)决定是否换实现；当前代码已经清楚且满足要求时，可以停止迁移。各框架与领域章节没有全部接入同一个应用。

## 每次改动留下验证证据

从仓库根目录运行独立项目测试：

```bash
python reference_code/evidence_assistant/run_tests.py
```

保存输入、预期、实际结果、配置与判定理由。主应用与基础题各自维护成功和失败用例，改动共享模块后两者都要回归。独立目录可以单独复制后用 `python run.py` 运行，或用 `python run_tests.py` 测试，具体入口见项目 `README.md`。

向量记忆、真实框架循环和状态图属于 `examples/` 的额外集成路线：

```bash
pip install -r examples/requirements.txt
python -m pytest examples/integration_tests -q
```

这些测试与离线项目范围不同。实际 ChromaDB、LangChain、LangGraph 集成并不自动证明调研应用的模型事实性、检索召回或成本；这些需要另设代表性任务集。继续在[第18章](./chapter_20_evaluation/README.md)整理回归门禁，在[第19章](./chapter_21_security/README.md)扩展权限与信任检查，最后按[第20章](./chapter_22_deployment/README.md)部署。
