# Chapter 11: Self-Evolving Agents

> 🧬 *"An ordinary Agent solves tasks; a Self-Evolution Agent solves tasks and learns how to better solve the next class of tasks."*

In earlier chapters, we taught the Agent to "use tools, remember, think, retrieve, and train". But these capabilities are mostly **static**: a hand-written Prompt will not improve on its own, a trained model's knowledge is frozen, and a lesson taught today may be repeated as a mistake tomorrow.

This chapter focuses on a theme that runs through the entire book — **Self-Evolution**: enabling the Agent to crystallize the by-products of every interaction (success patterns, failure lessons, transferable skills) and automatically reuse and improve them in subsequent tasks.

This chapter breaks "self-evolution" down from shallow to deep into four levels:

| Section | Content | Key Takeaway |
|---------|---------|--------------|
| 11.1 | Automatic Prompt Optimization | From handcrafted prompts to automatic search/reflection evolution: APE, OPRO, DSPy, TextGrad, GEPA |
| 11.2 | Self-Evolution Agent | Self-improvement closed-loop architecture, engineering deployment path, frontier research panorama, the three-role perspective, and the underrated "summarizer" |
| 11.3 | Agentic Data Flywheel | A continuous self-reinforcing loop of trajectory collection → quality filtering → reward labeling → training iteration |

**Progression path**: evolve "prompts" → evolve "the entire system behavior and frontier capabilities" → build a "continuous data flywheel". From Prompt-level evolution in 11.1, to the system-level closed loop and research frontier in 11.2, to the engineered continuous iteration system in 11.3 — each layer goes deeper.

---

*Start learning: [11.1 Automatic Prompt Optimization](./01_automatic_prompt_optimization.md)*

*Previous chapter: [Chapter 10: Agentic-RL: Reinforcement Learning for Agents](../chapter_agentic_rl/README.md)*

*Next chapter: [Chapter 12: LangChain In-Depth](../chapter_langchain/README.md)*
