# 从一次实际运行读懂证据型助手

日期：2026-10-22。以下输出来自当前标准库实现与随项目提供的虚构资料，不调用模型或外部服务。本页选一题讲清数据流；默认三题和自定义资料的运行方法见 [README](README.md)，接口边界见 [CONTRACTS](CONTRACTS.md)。

## 先运行，再看字段

在仓库根目录执行：

```bash
python -m reference_code.evidence_assistant --question '上线验收需要满足哪些条件？' --format json
```

独立复制整个目录时，在该目录执行 `python run.py --question '上线验收需要满足哪些条件？' --format json`。这里使用默认 `sample_data/`，CLI 自动载入 `sample_task_specs.json`；直接实例化 `ResearchAssistant` 时，调用方需要显式传入任务规格或独立验证器。

下面是同一次运行的**字段摘录**：为便于阅读，将首题放在 `finding` 下，并省略答案全文、原始证据及完整轨迹。实际 JSON 使用 `findings` 数组；其余数值与运行结果相同。

```json
{
  "status": "completed",
  "accepted": true,
  "finding": {
    "question": "上线验收需要满足哪些条件？",
    "status": "answered",
    "retrieval": {
      "status": "succeeded",
      "hits": [
        {
          "source": "02_acceptance.md:p2",
          "score": 3,
          "matched_terms": [
            "上线",
            "线验",
            "验收"
          ]
        },
        {
          "source": "02_acceptance.md:p3",
          "score": 3,
          "matched_terms": [
            "上线",
            "线验",
            "验收"
          ]
        }
      ]
    },
    "context": {
      "input_bytes": 847,
      "limit_bytes": 8192,
      "required_bytes": 847,
      "dropped": [],
      "compressed": []
    },
    "verification": {
      "status": "passed",
      "checks": [
        "original_source_excerpts",
        "lexical_relevance"
      ],
      "code": "supported_excerpts",
      "message": "各行摘录均可回查原文。",
      "sources": [
        "02_acceptance.md:p2",
        "02_acceptance.md:p3"
      ]
    },
    "coverage": {
      "status": "passed",
      "matched_points": [
        "资料负责人确认发布清单",
        "20条固定问题中至少18条返回正确来源",
        "所有无证据样例明确标记缺少证据",
        "任一条件未通过就不能完成验收"
      ],
      "missing_points": [],
      "checks": [
        "literal_task_requirements"
      ]
    }
  },
  "summary": {
    "questions": 1,
    "answered": 1,
    "unfinished": 0
  },
  "budgets": {
    "steps": {
      "used": 2,
      "limit": 8
    },
    "tool_calls": {
      "used": 2,
      "limit": 8
    },
    "context_bytes": {
      "peak_used": 847,
      "limit": 8192
    }
  },
  "failure": {}
}
```

`score=3`来自重合词项数量，不是概率或模型置信度；`线验`是中文二元词片，保留它是为了看清当前分词基线的行为。当前检索会过滤若干泛化短语，多词项中文问题要求至少两个词项匹配，但这不等于语义检索或保证召回。

`input_bytes=847`包含系统消息、两段来源资料与问题的实际UTF-8序列化开销。它是字节预算，不是模型token数，也不是只计算原文长度。本题保留全部必需证据，`dropped`与`compressed`均为空。

## 按数据流阅读四处代码

| 顺序 | 实际动作与字段 | 阅读位置 | 直接相关的回归 |
|---|---|---|---|
| 1 | 读取顶层资料，以文件名和段落产生稳定来源；`rank`返回分数和匹配词，`retrieval.hits`记录候选 | `research.py:load_documents`、`rag.py:DocumentQAAgent.rank` | `test_rank_explains_scores_and_preserves_retrieve_interface`、`test_independent_contrastive_cases` |
| 2 | 计划为每题安排检索与回答，二者都经Harness派发；上下文实际传给回答器 | `research.py:ResearchAssistant.run`、`context.py:GSSCPipeline`、`harness.py` | `test_constructed_context_is_consumed_and_original_evidence_rechecked`、两项预算测试 |
| 3 | `extract_answer`从消息提取原文，宿主逐行回查原始证据；结果在`verification`中 | `research.py:extract_answer/check_excerpts` | `test_correct_label_with_unsupported_claim_fails`、`test_context_removal_does_not_silently_read_original_corpus` |
| 4 | 独立任务规格核对必答要点；若同时配置验证器，二者均须通过；最终报告再核对执行与各题状态 | `task_acceptance.py:check_question_coverage`、`research.py:accept_report` | `test_specification_and_independent_verifier_both_pass`、`test_final_report_verification_rejection_is_reported` |

本题答案是`02_acceptance.md:p2`与`:p3`的原文，不是模型整理的推论。`verification=passed`说明摘录可在本次检索的原始证据中找到，且有词项重合；`coverage=passed`说明答案正文包含独立规格要求的四个字面要点。来源标签不能用来补足要点。

二者没有证明资料真实、全部业务要求都被规格列出、重写的自然语言语义等价或没有矛盾。更换任务时，先确定独立预期，再运行候选；不要从候选答案反推一个必然通过的规格。

## 故意让它无法完成

以下状态同样来自实际运行，而非模拟状态名：

| 输入/配置 | 题目状态 | 原文支撑 | 问题覆盖 | accepted | 首个未完成原因 |
|---|---|---|---|---|---|
| 默认验收问题 | `answered` | `passed` | `passed` | `true` | `—` |
| 同题但不配置任务要求 | `evidence_found` | `passed` | `unknown` | `false` | `coverage_unknown` |
| 生日无答案但命中同主题 | `evidence_found` | `passed` | `unknown` | `false` | `coverage_unknown` |
| 火星移民无检索命中 | `no_evidence` | `not_run` | `not_checked` | `false` | `no_evidence` |
| 默认问题仅允许1步 | `retrieved` | `not_run` | `not_checked` | `false` | `step_budget_exhausted` |

无任务规格时，即使回答摘录有效，也仅为`evidence_found`；执行计划完成仍不能将报告标为成功。生日问题仍可因“青禾项目负责人”命中同主题资料，原文核对也可通过，但摘录没有给出生日，且没有该问题的任务规格，所以覆盖保持unknown、报告不被验收。这正是“有出处”不能替代“回答了问题”的反例。火星移民问题没有达到检索阈值的命中，回答器不执行；这两种失败必须分开。它们不代表对所有无答案问题都有完美识别能力。

一步预算仅够检索，尚未执行回答。报告同时保留`execution.trace`、`tool_trace`、`failure`与`budgets`，读者可以区分没有证据、生成失败、要点缺失和执行预算耗尽。CLI仅在报告`accepted=true`时退出0，其余情况退出1；Markdown输出也显示首个失败步骤和缺失要点。

继续练习：用合法摘录回答，但故意只保留第一个要点。原文支撑应通过，覆盖应失败，题目状态应为`partial`。对应测试是`test_valid_excerpts_missing_task_points_remain_partial_even_if_verifier_accepts`。不要删改规格来掩盖遗漏。

## 运行对应测试

```bash
python reference_code/evidence_assistant/run_tests.py
```

项目测试包含基础六阶段、检索与任务要求、调研报告的正常及失败路径；详见`tests/test_retrieval_and_tasks.py`和`tests/test_research.py`。测试替身证明接口与边界行为，不证明线上模型质量或生产可用性。
