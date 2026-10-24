# Chapter 25: Self-Evolving Agents

**Reading level**: Research extension, after an application and evaluation baseline; model training is not required.

## This chapter's iteration

- **Previous limitation**: The application has acceptance criteria, but prompt and skill changes still rely on manual trial and error. Higher scores on familiar examples may reflect overfitting or weakened permission constraints.
- **This chapter adds**: Turn failure traces into candidate changes while separating proposal, fixed evaluation, promotion, and rollback. Limit what may change and preserve independent tests and execution boundaries.
- **How to verify**: The existing `examples/auto_research_demo/main.py` uses standard-library candidate search on synthetic classification, exposing configurations, scores, and keep/reject records. It does not rewrite Agent prompts or skills, and its evaluator separation is organizational rather than enforced isolation. Reader extension: connect your project and independent tests; verify that regressions are rejected, rollback works, and candidates cannot edit tests or permissions before claiming gains.
- **Next step**: Return accepted candidates to [chapter 18's regression gate](../chapter_20_evaluation/README.md), then [chapter 19's boundary checks](../chapter_21_security/README.md) and [chapter 20's release process](../chapter_22_deployment/README.md). Choose [chapter 24](../chapter_agentic_rl/README.md) when the bottleneck calls for weight changes; training is not required to improve prompts or skills.

> 🧬 *"An ordinary Agent solves tasks; a Self-Evolution Agent solves tasks and learns how to better solve the next class of tasks."*

The preceding chapters establish applications that execute tasks, check results, and can be deployed. The next question is how to use recurring failures to improve the next version without weakening tests or execution boundaries. This chapter can follow a capstone directly; chapter 24's weight training is optional.

This chapter focuses on a theme that runs through the entire book — **Self-Evolution**: enabling the Agent to crystallize the by-products of every interaction (success patterns, failure lessons, transferable skills) and automatically reuse and improve them in subsequent tasks.

This chapter organizes self-evolution into three levels:

| Section | Content | Key Takeaway |
|---------|---------|--------------|
| 25.1 | Automatic Prompt Optimization | From handcrafted prompts to automatic search/reflection evolution: APE, OPRO, DSPy, TextGrad, GEPA |
| 25.2 | Self-Evolution Agent | Self-improvement closed-loop architecture, engineering deployment path, frontier research panorama, the three-role perspective, and the underrated "summarizer" |
| 25.3 | Agentic Data Flywheel | A continuous self-reinforcing loop of trajectory collection → quality filtering → reward labeling → training iteration |

**Progression path**: evolve "prompts" → evolve "the entire system behavior and frontier capabilities" → build a "continuous data flywheel". From Prompt-level evolution in 25.1, to the system-level closed loop and research frontier in 25.2, to the engineered continuous iteration system in 25.3 — each layer goes deeper.

---

*Start learning: [25.1 Automatic Prompt Optimization](./01_automatic_prompt_optimization.md)*

*Previous chapter: [Chapter 24: Agentic-RL: Reinforcement Learning for Agents](../chapter_agentic_rl/README.md)*

*Return to acceptance: [Chapter 18: Agent Evaluation and Optimization](../chapter_20_evaluation/README.md)*
