# 参考项目的数据与完成契约

本页约定“产物已生成”和“产物已验收”的区别。调研应用的任务覆盖规则位于 `task_acceptance.py`，原文支撑规则位于 `research.py`；距离计算基础练习的规则位于 `contracts.py`、`acceptance.py`。相应反例都在项目的 `tests/` 中。

## 本地资料调研助手

主应用位于 `research.py`，入口是 `python run.py`（本目录）或 `python -m reference_code.evidence_assistant`（仓库根目录）。测试位于 `tests/test_research.py`、`tests/test_retrieval_and_tasks.py`。以下约定适用于调研应用；后面的距离计算契约属于更小的基础练习。

| 输入/产物 | 当前约定 |
|---|---|
| 本地资料 | 指定目录顶层的1–32份UTF-8 `.md` / `.txt`，每份最多64 KiB；不递归目录，不读取符号链接 |
| 来源 | 文本按空行切段，标签为`文件名:p段号`；Markdown标题不当作正文，文件名不能含方括号或换行 |
| 问题 | 1–4个非空问题，默认使用随附资料的3个问题；每题都建立检索与回答两个步骤 |
| 检索 | `rank`按不同匹配词项数量排序，每题最多3段；`retrieval.hits`保留来源、`score`与`matched_terms`。中文通用问句短语先被过滤，再形成相邻双字片段，不是真正分词；含多个中文片段的查询，只有中文重合时至少需两个匹配；单片段/英文查询可凭一个词项命中。同分保留输入顺序 |
| 回答 | 默认离线摘录，逐行为`[source] 原文片段`；回答器只消费构造后的上下文 |
| 原文支撑 `verification` | 每行来源属于本题命中，摘录能在规范化原文段落中找到，且与问题存在词项重合；不能据此认定题目已回答完整 |
| 任务覆盖 `coverage` | 单独与匹配的任务规格和/或独立`verifier`比较；没有匹配规格或验证器时为`unknown` |
| 报告 | 每题保留`status`、`stage`、`reason`、答案、来源、证据、检索解释、上下文计量、原文支撑与覆盖；顶层保留`summary`、`budgets`、`failure`、计划与工具轨迹 |

检索分数是词项计数，不是置信度、准确率或完整回答概率。命中同主题段落不意味着其中有答案。例如负责人生日问题可能命中负责人相关资料，`verification`也可能通过，但覆盖仍未知；与完全无命中的`no_evidence`不同。

### 默认15要点与自定义任务规格

CLI使用包内 `sample_data/` 时，自动加载 `sample_task_specs.json`：上线验收4个要点、工具预算/超时6个、回退5个，共15个。规格独立于资料、检索和回答器，只用于验收，不是提供给生成器的证据。显式 `--task-spec FILE` 可指定另一个任务规格。复制资料到自定义目录不会自动沿用默认规格；直接实例化 `ResearchAssistant` 也不会自动加载，调用方需传入 `task_specs=load_task_specs(path)` 或独立验证器。

自定义JSON的顶层字段必须为 `schema_version`（整数1）和 `tasks`。最小形态如下；实际使用时还要准备独立资料和对应问题：

```json
{
  "schema_version": 1,
  "tasks": [{
    "question": "上线验收需要满足哪些条件？",
    "requirements": [{
      "id": "regression",
      "description": "完成回归测试",
      "all_of": ["完成回归测试"]
    }]
  }]
}
```

`tasks`允许1–64个规格；每个规格仅有`question`和`requirements`。问题去除空白并忽略大小写后匹配，不能自动把不同标点或同义问法当成同一题。每题1–32个要点，字段严格为`id`、`description`、`all_of`，ID在本题内唯一；`all_of`含1–16个非空字面片段，必须全部出现在回答正文中。行首来源标签会移除后再检查，避免把文件名当成回答要点。

规格由任务维护者在生成回答前独立制定、评审和冻结。源码不会从当前回答反推要求，也不会把规格加入证据。每个要点返回`details`，列出`missing_fragments`；`matched_points`与`missing_points`方便定位遗漏。空白/大小写归一化后的字面包含检查适用于本例冻结摘录任务，不构成通用语义证明：它不会可靠判断同义改写、否定关系、矛盾、要点顺序或资料真实性。

Python调用方可提供 `verifier(question, answer, documents)`，其中`documents`为本题命中的原文元组，必须返回严格的`True`才通过。同时配置匹配规格与验证器时，两者都要通过；验证器返回True不能覆盖缺失的规格要点。无匹配规格时，显式独立验证器可建立任务判定；未配置两者则保持`unknown`。CLI只接受`--task-spec`，不提供验证器参数。

### 状态、退出码与定位

`status=completed`且`accepted=true`要求计划执行完成，且每题的`verification.status`和`coverage.status`都为`passed`。自定义资料没有匹配规格或独立验证器时，即使所有摘录有原文支撑，单题也仅为`evidence_found`，`coverage.status=unknown`，整体`incomplete`、退出码`1`。单看`execution.status=completed`不足以交付报告。

| 单题情况 | 关键字段与定位 |
|---|---|
| 两层验收均通过 | `status=answered`、`stage=completed` |
| 有有效摘录，未配置本题要求 | `status=evidence_found`、`coverage.status=unknown`、`reason.code=coverage_unknown` |
| 有效摘录缺必答点，或独立验证器拒绝/异常 | `status=partial`、`coverage.status=failed`；查看`missing_points`、`details`及原因码 |
| 检索没有合格命中 | `status=no_evidence`、`verification.status=not_run`、`coverage.status=not_checked`；回答器未执行 |
| 检索、生成或原文支撑失败 | `retrieval_failed`、`generation_failed`或`verification_failed`，`stage`、`reason.code/message`保留位置；覆盖尚未检查时不能称为已通过 |
| 必需上下文装不下 | `status=context_budget_exhausted`；核对`context.required_bytes`与`limit_bytes` |
| 步骤或调用预算耗尽、工具未授权 | 逐题可能停在`not_started`或`retrieved`；顶层`failure`记录`step_budget_exhausted`、`tool_budget_exhausted`或`tool_not_authorized` |

顶层`summary`统计题数、已验收与未完成数；`budgets`记录步骤、调用记录与上下文用量；`failure.question_index`从0开始，配合`step`、`code`、`message`定位首个未完成位置，报告级失败的索引可为null。完整细节仍在`findings`、`execution.trace`和`tool_trace`中。

任一题未通过或预算耗尽，整体均为`incomplete`、`accepted=false`。输入/规格非法或输出文件写入失败时，CLI向stderr解释原因并以非零状态退出。

默认最多8个计划步骤、8次工具调用，每题输入消息最多8192 UTF-8字节。检索与回答都经过Harness注册表；耗尽或未授权时不再执行工具。上下文的必需证据超限时直接拒绝，不丢掉证据后伪装为完整回答。次数和字节预算不等于调用耗时限制；注入外部回答器时，其超时和模型质量需要另外配置与验证。

只有全部问题验收通过，CLI退出码才是`0`。`--format json`保留详细状态，默认Markdown便于阅读。`--output`只创建新文件，不覆盖已有文件；未传该参数时只打印结果。未通过的报告仍可作为失败记录保存，必须连同其未完成状态读取。

原文支撑检查与任务覆盖是两层不同的证据。字面要点通过说明冻结的要求已出现，不证明开放式问题的语义完整性。真实模型生成摘要、跨资料推断与业务结论，需要代表性任务集与适当的独立验收。

## 六阶段基础练习的数据链

调用方提供 `Document(source, text)`。系统从实际进入检索结果和上下文的原文提取数量，统一为米后计算光行时间；记忆中读出的输出单位参与最终换算。没有有效数量时不运行计算工具。Harness 阶段的提取与计算都经过工具注册表，并分别消耗一次调用预算。

| 产物 | 必需字段 | 约束 |
|---|---|---|
| 原始资料 | `source`, `text` | 1–3 份，来源唯一；来源为 1–64 个英文字母、数字、下划线或连字符；每份正文 1–4000 UTF-8 字节 |
| `provenance` | `source`, `text`, `sha256` | 保留实际消费的原文与摘要，最终必须与调用方输入一一对应 |
| `quantity` | `value`, `unit`, `source`, `quote`, `evidence_sha256` | 十进制数值保留字符串；单位只接受 `m`、`km`；引文和摘要可回查 |
| `calculation` | `distance_metres`, `speed_metres_per_second`, `seconds`, `output_value`, `output_unit` | 光速固定为 299792458 m/s；输出单位为 `seconds` 或 `milliseconds` |
| 成功产物 | `status`, `accepted`, `acceptance`, `output`, `answer` | 独立核对原文、数量、单位、运算、最终回答及引用 |

教学输入严格接受 `Moon distance is 100 km.`、`Moon mean distance in this teaching fixture is 384400 km.` 或 `Distance: 100000 m.` 这样的单句，也接受 `Moon mean distance is ...` 与 `Moon distance in this teaching fixture is ...`。大小写不限；数值使用 ASCII 十进制，最多 24 位数字（包括前后零），不支持科学计数法；换算后的范围为 `[1e-12, 1e18]` 米。多份资料必须在单位换算后完全一致。其他自然语言、单位与数值范围需要明确的适配器，不能静默猜测。

验收器重新读取调用方原始输入，使用独立的解析与 Decimal 运算，不调用生产端的提取器、换算器或计算器。数值采用 `1e-12` 相对误差，不用固定绝对容差放行接近零的错误答案。来源摘要只检查数据链的一致性，不证明现实资料的真实性。

`reference_code/evidence_assistant/tests/fixtures/light_travel_cases.json` 保存独立预期，用于回归测试，运行时不读取它。测试还故意破坏提取单位、计算器、来源摘要、上下文和最终答案，确认验收会拒绝。

## 基础练习与关联章节的状态

| 组件 | 完成判据 | 失败或停止 |
|---|---|---|
| 六阶段基础练习 | `status=completed` 且 `accepted=true` | `failed` 为输入或执行失败；`budget_exhausted` 为步骤、工具或上下文预算耗尽。已生成的错误产物可为 `completed`，但必须 `accepted=false` |
| 文档问答 | `answered` 且检索、生成、验证阶段均通过 | `no_evidence`、`retrieval_failed`、`generation_failed`、`verification_failed`；空回答及缺失/未知引用不能通过 |
| 章节用 `research_agent.py` | 所有问题的简化问答验收通过，且计划完成 | 任一问题不通过或步骤用尽即 `incomplete`；主应用另适用上面的`coverage`契约 |
| 查询重试 | `answer_accepted(result)` 为真 | 仅无证据或暂时性调用错误可重试；受 `max_queries` 限制，保留最后一次失败 |
| 开发团队 | `delivered`：实际测试通过且运维、文档写入完成 | `failed`、`timed_out`、`budget_exhausted`；失败保存报告，不进入成功交付路由 |

问答默认只核对引用标签，不证明结论受到原文语义支持。需要通过 `verifier(question, answer, documents)` 注入独立任务判据。开发团队的默认开发者同时生成代码与测试，因此 `delivered` 也不证明任意业务需求已独立验收；真实需求必须配置固定的外部验收集。

六阶段基础练习在未运行独立验收时返回 `acceptance.status=not_run`，失败信息为 `error.code` 与 `error.message`；没有计算结果时不伪造 `output` 或 `answer`。开发团队的 `workspace_dir` 必须不存在或为空，复用非空目录会在写入前被拒绝，避免旧交付物与本轮失败实现混合。

开发团队默认最多生成 3 版、每次测试最多 30 秒。macOS/Linux 在成功、失败、超时后均清理整个测试进程组并回收主进程。该机制不能隔离主动逃离会话的恶意代码；模型生成调用时限和隔离执行器是独立的配置。

## 基础练习的替换输入

在仓库根目录运行：

```bash
printf 'Moon distance is 100 km.\n' > /tmp/light-distance.txt
python -m reference_code.evidence_assistant.demo --all --file /tmp/light-distance.txt --output-unit milliseconds
printf 'Moon distance is unknown.\n' > /tmp/light-distance.txt
python -m reference_code.evidence_assistant.demo --stage harness --file /tmp/light-distance.txt
```

第一条运行输出约 `0.33356409519815205` 毫秒，退出码为 `0`；第二条返回 `invalid_quantity`、`accepted=false`，退出码为 `1`。不加 `--file` 使用虚构的默认月球资料。验证范围与实际结果见 `examples/VALIDATION.md`。

## English summary

For the research application, `verification` checks quoted evidence support and `coverage` checks independent task requirements. Bundled documents automatically select three frozen questions with 15 requirements in `sample_task_specs.json`. Custom documents require `--task-spec FILE` or an explicitly injected Python verifier. Without matching requirements, supported excerpts remain `evidence_found` / coverage `unknown`, with `accepted=false` and exit code 1. Missing configured points produce `partial`; retrieval, generation, evidence and budget failures retain stage and reason fields. Retrieval scores count lexical matches, not confidence. Task specifications are never used as evidence. Literal fragments do not establish general semantic completeness or source truth.

The shared project consumes evidence through extraction, unit normalization, calculation and an independent verifier. Caller evidence has unique source IDs and hashes; quantities retain their original value, unit and quote. Input is a controlled, single-statement decimal fixture with at most 24 digits, normalized to `[1e-12, 1e18]` metres. Source hashes check consistency, not authenticity. The verifier re-parses caller inputs without invoking the producer's parser or calculator and checks the final answer as well as structured fields. Test expectations are stored separately and are never loaded by the execution path.

An artifact is usable only when `status=completed` **and** `accepted=true`. Empty evidence, unsupported units, contradictions and exhausted budgets fail explicitly. QA citation-label checks do not establish semantic support; inject a task verifier for that. Developer-team delivery means its generated tests and artifact writes passed; independent business acceptance and production isolation remain separate requirements.
