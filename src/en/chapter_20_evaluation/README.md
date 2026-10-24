# Chapter 18: Agent Evaluation and Optimization

**Reading level**: Core; revisit from the first hands-on exercise.

## This chapter's iteration

- **Previous limitation**: A few chapter cases establish local behavior, but do not show whether a change regresses, improves real tasks or stays within cost.
- **This chapter adds**: Version the earlier task cards, failures and traces as a regression set with independent criteria; report quality, refusals, execution failures, latency and call costs separately.
- **How to verify**: Start with the fixed expectations and corrupted artifacts in `reference_code/evidence_assistant/tests/test_project_contracts.py` and `reference-agent/tests/test_evaluation.py`. Keep an independent held-out set for your real tasks and compare identical inputs and configuration. Candidate-generated scores cannot authorize release. Offline tests do not measure live-model gains.
- **Next step**: Take failure categories and release gates to [chapter 19](../chapter_21_security/README.md). No particular framework is required; revisit evaluation from chapter 3 onward.

> How do you know if your Agent is "good" or "not good"? Evaluation and optimization are the key to going from "usable" to "excellent."

---

## Establish a Baseline from the First Exercise

When working through chapter 3, save fixed inputs and expected results, record tool traces, and include a failure case. Rerun those cases whenever you add memory, retrieval, or planning, then add cases exposing the new capability's weaknesses. A model's self-score is not an acceptance verdict.

For an offline starting point, install `pip install -e ".[dev]"` inside `reference-agent/` and run `python -m pytest tests/test_evaluation.py -q`. Its simple matching on fixed samples illustrates a regression gate. Evaluate real-task correctness, evidence, tool behavior, cost, and failure states separately. Keep the input, expectation, result, verdict rationale, and configuration version for each case.

## Chapter Overview

Agent development doesn't end when you finish writing the code — you need to measure its performance, identify weaknesses, and continuously improve. This chapter introduces systematic evaluation methods, benchmarks, prompt tuning techniques, cost control strategies, and observability systems.

## Chapter Goals

- ✅ Master the core methods of Agent evaluation (rule-based, LLM-based, human evaluation)
- ✅ Understand commonly used industry benchmarks and evaluation metrics
- ✅ Learn a systematic prompt tuning process
- ✅ Implement cost control and performance optimization
- ✅ Build a comprehensive observability system for Agents

## Chapter Structure

| Section |
|---|
| [18.1 How to Evaluate Agent Performance?](./01_evaluation_methods.md) |
| [18.2 Benchmarks and Evaluation Metrics](./02_benchmarks.md) |
| [18.3 Prompt Tuning Strategies](./03_prompt_tuning.md) |
| [18.4 Cost Control and Performance Optimization](./04_cost_optimization.md) |
| [18.5 Observability: Logging, Tracing, and Monitoring](./05_observability.md) |
| [18.6 Agent-Specific Evaluation Frameworks](./06_agent_evaluation.md) |
| [18.7 A/B Testing and Regression Test Automation](./07_ab_testing.md) |
| [18.8 Model Routing Evaluation](./08_model_routing.md) |

## 🔗 Learning Path

Start with the first fixed case in [chapter 3](../chapter_tools/README.md); no framework prerequisite.

Take failure categories and release gates to [chapter 19](../chapter_21_security/README.md). No particular framework is required; revisit evaluation from chapter 3 onward.

---

*Next section: [18.1 How to Evaluate Agent Performance?](./01_evaluation_methods.md)*
