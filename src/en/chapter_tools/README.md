# Chapter 3: Tool Calling (Tool Use / Function Calling)

**Reading level**: Core.

## This chapter's iteration

- **Previous limitation**: A model may explain light travel time but confuse units, merely claim a tool ran, or report success after a tool error.
- **This chapter adds**: Connect tool schemas, host argument/permission checks, execution, call-ID correspondence, and independent acceptance in an observable local search/calculation loop.
- **How to verify**: Run `python -m reference_code.tool_calling` and `python -m unittest examples.integration_tests.test_tool_calling_chapter -v`. Success requires three fixed-provider turns, two actual tool proposals processed, and the frozen rubric passing. Restrict budgets, deny permission, and corrupt results to preserve incomplete outcomes. The main path separately offers `python -m reference_code.evidence_assistant.demo --stage tools` for evidence/unit contracts.
- **Next step**: Retain explicitly consented unit preferences in [Chapter 4](../chapter_memory/README.md). Correct arithmetic does not establish correct preferences in the next session.

Tool calling submits model proposals to a host program. A model may select tools and fill arguments; the application still decides authority, validation, failure representation, and sufficient evidence of completion. Tools do not grant arbitrary system access.

## Goals and Scope

A moderately sized local task connects protocol and engineering boundaries: retrieve frozen teaching material about Moon distance, calculate seconds through bounded arithmetic, then independently check evidence, value, unit, and source. The fixed provider consumes actual tool results; it does not simulate natural-language decision quality.

The companion is `reference_code/tool_calling/`, run from the repository root. It reuses `examples/chapter03/safe_math.py` and the main project's `rag.py`; copying this directory alone is not a complete standalone product. Defaults use the standard library, without web search, arbitrary file access, or online models. SDK exercises use pinned OpenAI 2.54.0 types and an HTTP mock in the repository integration environment.

| Section | Increment | Deliverable |
|---|---|---|
| [3.1 Why tools](./01_why_tools.md) | Separate proposals, observations, and acceptance | Tool-boundary table |
| [3.2 Function Calling](./02_function_calling.md) | Schema and call-ID message exchange | Correctly correlated message trace |
| [3.3 Custom tools](./03_custom_tools.md) | Arguments, permissions, results, budgets | Rejection before execution |
| [3.4 Descriptions](./04_tool_description.md) | Align text with actual contracts | Independent tool-selection cases |
| [3.5 Search/calculation](./05_practice_search_calc.md) | Dependencies, failure stops, independent rubric | Success and incomplete reports |
| [3.6 Papers](./06_paper_readings.md) | Optional research route | Compare research tasks with local mechanisms |

Complete local positive and negative cases before choosing real models, web search, or more tools. Fixed-fixture tests establish neither model selection quality, external source truth, nor production readiness.

Prerequisites: [Chapter 2](../chapter_llm/README.md), [setup](../chapter_setup/README.md), Python functions and JSON. Continue to [Chapter 17](../chapter_19_protocol/README.md) for standardized protocols and [Chapter 19](../chapter_21_security/README.md) for authorization/isolation.

*Next: [3.1 Why Do Agents Need Tools?](./01_why_tools.md)*
