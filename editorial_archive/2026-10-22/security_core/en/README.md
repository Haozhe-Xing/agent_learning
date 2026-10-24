# Chapter 19: Security and Reliability

**Reading level**: Core; revisit from the first tool call.

## This chapter's iteration

- **Previous limitation**: Passing normal tasks does not show that hostile sources cannot alter instructions, cross-user memory cannot leak, or dangerous calls are denied before execution.
- **This chapter adds**: Separate model, source, identity and tool-service trust boundaries. Put permissions, sensitive-data handling and refusal paths in host code.
- **How to verify**: Supplied `reference-agent/tests/test_security.py` and `reference_code/evidence_assistant/tests/test_learning_path.py` check basic input rules, memory isolation and denied calls never reaching execution. Readers must test injection, cross-user access, privilege violations and log leakage against their actual backend. Filters and process groups do not replace an isolated executor.
- **Next step**: Bring the permission matrix, attack records and coverage limits to [deployment](../chapter_22_deployment/README.md). Use basic checks from the first tool call.

> Agents have tools and autonomy — security becomes especially important. This chapter is the key to going from "usable" to "trustworthy."

---

## Check Execution Boundaries from the First Tool Call

Safety checks accompany the earlier exercises: tool and calculation budgets in chapter 3; consent, deletion, and user isolation in chapter 4; source data versus instructions in chapter 6; backend authorization and stopping conditions in chapters 10–11.

For each exercise, record data sources, allowed actions, the code enforcing permissions, and rejection or budget-exhaustion behavior. Test both an ordinary input and an unauthorized request. A prohibition in a prompt does not establish backend enforcement. After installing the development dependencies in `reference-agent/`, run `python -m pytest tests/test_security.py -q` for the supplied basic rules; these cases do not constitute production security validation.

## Chapter Overview

Agent security and reliability involves five aspects: prompt injection defense, hallucination control, permission management, data protection, and behavioral alignment. This chapter explains the principles and practical defense solutions for each aspect.

## Chapter Goals

- ✅ Understand the principles and defense strategies for prompt injection
- ✅ Master techniques to reduce hallucinations and improve factuality
- ✅ Design a least-privilege system and code execution sandbox
- ✅ Implement sensitive data detection and desensitization
- ✅ Build behavioral boundaries and rejection policies

## Chapter Structure

| Section |
|---|
| [19.1 Prompt Injection Attacks and Defenses](./01_prompt_injection.md) |
| [19.2 Hallucination Problems and Factuality Assurance](./02_hallucination.md) |
| [19.3 Permission Control and Sandbox Isolation](./03_permission_sandbox.md) |
| [19.4 Sensitive Data Protection](./04_data_protection.md) |
| [19.5 Controllability and Alignment of Agent Behavior](./05_alignment.md) |
| [19.6 Paper Reading: Frontiers in Security and Reliability](./06_paper_readings.md) |
| [19.7 Guardrails Runtime Protection](./07_guardrails_runtime.md) |
| [19.8 Red Teaming Methodology](./08_red_teaming.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Bring the permission matrix, attack records and coverage limits to [deployment](../chapter_22_deployment/README.md). Use basic checks from the first tool call.

---

*Next section: [19.1 Prompt Injection Attacks and Defense](./01_prompt_injection.md)*
