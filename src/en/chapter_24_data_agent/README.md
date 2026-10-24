# Chapter 22: Project Practice: Intelligent Data Analysis Agent

**Reading level**: Capstone; choose at least one domain from chapters 21–23.

## This chapter's iteration

- **Previous limitation**: Correct units in an evidence task do not establish database authorization, consistent analysis definitions or report-to-source reconciliation.
- **This chapter adds**: Transfer the evidence and calculation contract to read-only connections, query plans, definitions, source reconciliation and report acceptance.
- **How to verify**: Section 22.2 supplies real SQLite boundaries and a SQL-proposal interface; analysis, charts, and reports still need assembly and acceptance. Reconcile SQL, aggregates and report totals on fixed synthetic data. Writes, empty results, missing columns and duplicates must be denied or explicitly reported; enforce database permissions in the backend.
- **Next step**: A passing domain task completes the application route. [Coding](../chapter_23_coding_agent/README.md) and [multimodal](../chapter_25_multimodal/README.md) are alternatives; enter [research](../part7.md) only for a measured limitation.

> Data analysis with natural language — from "help me analyze last month's sales" to a complete analysis report.

---

## Chapter Overview

Data analysis is one of the most practical application scenarios for Agents. This chapter separates natural-language proposals, database permissions, analysis, and report acceptance. It supplies a runnable SQLite boundary; the complete domain workflow remains a design and assembly exercise.

## Chapter Goals

- ✅ Separate SQL proposals, execution permissions, and business acceptance; run bounded SQLite queries
- ✅ Build automated data analysis and visualization
- ✅ Extract insights from data using LLM
- ✅ Automatically generate structured analysis reports
- ✅ Assemble one bounded analysis task and deliver a source-reconciled report with acceptance records

## Chapter Structure

| Section |
|---|
| [22.1 Requirements Analysis and Architecture Design](./01_requirements.md) |
| [22.2 Data Connection and Querying](./02_data_connection.md) |
| [22.3 Automated Analysis and Visualization](./03_analysis_visualization.md) |
| [22.4 Report Generation and Export](./04_report_generation.md) |
| [22.5 Integrated Design and Runnable Foundation](./05_full_implementation.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

A passing domain task completes the application route. [Coding](../chapter_23_coding_agent/README.md) and [multimodal](../chapter_25_multimodal/README.md) are alternatives; enter [research](../part7.md) only for a measured limitation.

---

*Next: [22.1 Requirements Analysis and Architecture Design](./01_requirements.md)*
