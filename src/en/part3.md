# Part III: Frameworks and Implementation Choices

Parts I–II established tools, memory, planning, evidence, context, and execution controls for a minimal Agent. This part addresses implementation choices: identify a concrete maintenance problem, then decide whether to adopt a development library, use a product, or borrow an architectural mechanism.

Chapters 10–11 cover development libraries as needed. Chapters 12–15 are optional product and architecture case studies. LangChain and LangGraph can be combined or studied separately; they are not a required stack for every project. Application development does not require the research material in chapters 24–25 first.

## Bring the Existing Artifacts

Keep the [continuing project's](./learning_project.md) task input, sources, tool permissions, call budget, execution trace, and independent acceptance results. When introducing another implementation, replace the part responsible for the current problem and rerun the same success and failure cases.

Repeated adapter code suggests investigating shared interfaces. Unsafe resumption after interruption suggests investigating explicit state and recovery. Keeping a working minimal implementation is also a valid final decision.

## Separate the Layers

| Layer | Learning objects in this part | Question to answer |
|---|---|---|
| Development library | LangChain, LangGraph | Which interfaces and execution state does the library manage, and which business constraints remain the application's responsibility? |
| Product | OpenClaw, Hermes, Claude Code | Does it fit the intended entry point and task? Do permissions, configuration, data handling, and failure records meet requirements? |
| Architecture case study | DeepSeek Harness and designs examined in the other products | Which interface or control mechanism is useful? Should it be adopted, or only studied? |

“Architecture case study” describes its teaching role. A product can also be studied for its architecture. Reading a case does not replace execution evidence or establish that several products can be connected directly.

## Choose a Chapter from a Problem

| Current problem | Layer and candidate chapter | Verify with a fixed task | Skip or stop when |
|---|---|---|---|
| Repeated model, tool, and message adapters | Library: [10, LangChain](./chapter_langchain/README.md) | Preserve inputs, permissions, and acceptance cases; compare call traces, error results, and adapter code | Existing interfaces are clear and migration only adds dependencies |
| Branching, pause, recovery, or human confirmation is hard to track | Library: [11, LangGraph](./chapter_langgraph/README.md) | Repeat the same interruption and failure inputs; inspect resume position, duplicate side effects, budgets, and terminal states | A short workflow needs no recovery and an explicit loop is sufficient |
| Use a personal assistant from chat | Product case: [12, OpenClaw](./chapter_openclaw/README.md) | Choose one target channel and verify identity, session boundaries, tool access, error messages, and task results | Only a backend is needed or channel and permission requirements cannot be met |
| Study cross-session memory and skill accumulation | Product/architecture case: [13, Hermes](./chapter_hermes/README.md) | Use held-out tasks to compare memory or skills enabled and disabled; record review, disabling, and rollback | There is no long-term use case or no independent way to judge improvement |
| Use a coding assistant or study code-operation boundaries | Product/architecture case: [14, Claude Code](./chapter_claude_code/README.md) | Fix a coding task and inspect the patch, actual tests, authorization, and failure records | The task is unrelated to coding or execution conditions cannot be reproduced |
| Study replaceable components and plugin lifecycle | Architecture case: [15, DeepSeek Harness](./chapter_deepseek_harness/README.md) | Pin a version and configuration, replace one component, then inspect interfaces, state compatibility, and failure cleanup | Only an assistant is needed or replacement benefits do not justify maintenance |

The table provides reading and experiment entry points. Stop a migration if it fails permission, data, or correctness requirements; retain the original implementation and record why. Completing the table does not require enabling every tool or connecting personal data.

## What the Supplied Examples Establish

| Material | Existing entry point | What can be observed | What it does not establish |
|---|---|---|---|
| Local-document research application | `python -m reference_code.evidence_assistant` | Multiple documents and questions, retrieval, planning, measured context, a cited report, and explicit failures | Open-web research, complete semantic acceptance, or product superiority |
| Foundation calculation exercise | `python -m reference_code.evidence_assistant.demo --all` | Six-stage data flow, units, permissions, budgets, and independent numeric acceptance; identified as `offline-fixture` | Live-model understanding or web-search quality |
| Library companions | `examples/chapter12/customer_service.py`, `examples/chapter13/content_workflow.py` | Tool orchestration and state-graph behavior with fixed model outputs; directory names retain historical chapter numbers | Comparable results across two different tasks or live-model quality |
| Product and architecture chapters | Case descriptions in chapters 12–15 | Analysis of entry points, state, permissions, and extensions, followed by a reproducibility plan | Four uniform offline product adapters or execution in an identical environment |

The library companions use different tasks. A valid implementation comparison requires porting the **same task input, expected result, and failure cases** into each candidate and running them separately. Successful demonstrations of unrelated tasks are insufficient. Record model configuration, calls, duration, and failures separately when using a live model.

## How the Chapters Connect

Chapter 10 focuses on interfaces and tool orchestration. Continue to chapter 11 when the next problem is where to resume, which actions must not repeat, or when to wait for confirmation. Readers who only need integration can stop after the relevant material.

Chapter 11 makes execution state inspectable. Chapters 12–15 provide optional design samples for chat entry points, cross-session accumulation, code operations, and component extension. Choose a relevant case, map its mechanism onto the existing assistant, and retain evidence of its effect.

After one candidate comparison, use [section 15.7](./chapter_deepseek_harness/07_summary_and_decision_matrix.md) to record the decision. Completion means explaining a choice with evidence and limits, rather than reading a certain number of products.

## Continue According to the Next Problem

- For independently assignable work, test whether collaboration improves outcomes in [chapter 16, Multi-Agent Collaboration](./chapter_18_multi_agent/README.md). Compare with one Agent first; additional roles alone are not a benefit.
- For interoperability across process or service boundaries, read [chapter 17, Agent Communication Protocols](./chapter_19_protocol/README.md). Judge protocol requirements separately from the number of Agents.
- To turn existing records into regression gates, continue with [18, Evaluation](./chapter_20_evaluation/README.md), [19, Security and Reliability](./chapter_21_security/README.md), and [20, Deployment](./chapter_22_deployment/README.md).
- With a defined task and acceptance criteria, choose a capstone in chapters 21–23: [coding](./chapter_23_coding_agent/README.md), [data analysis](./chapter_24_data_agent/README.md), or [multimodal work](./chapter_25_multimodal/README.md).

---

*Start with [chapter 10, LangChain](./chapter_langchain/README.md) for integration needs, or [chapter 11, LangGraph](./chapter_langgraph/README.md) for state needs.*
