# Chapter 15: DeepSeek Harness — The Everything-Is-a-Plugin Foundation

**Reading level**: Optional architecture case for component replacement and plugin management.

Study DeepSeek Harness as an architecture case for component boundaries and plugin lifecycle. Ask verifiable questions about model adapters, tools, state, and extensions, then decide which interfaces to borrow and which contracts a replacement must preserve.

## This chapter's iteration

- **Previous limitation**: Capabilities run, but replacement scope and cleanup responsibility are unclear when a model, tool, or state implementation changes.
- **This chapter adds**: Analysis of plugin, service, and event boundaries, including loading, calls, failure, and unloading in the lifecycle.
- **How to verify**: No uniform DeepSeek Harness adapter is supplied. Pin source or release version, configuration, and one replacement target; inspect identical inputs and outputs, errors, permissions, and cleanup. Unexecuted material remains architecture analysis.
- **Next step**: Record evidence in [section 15.7](./07_summary_and_decision_matrix.md). Continue to [chapter 16, Multi-Agent Collaboration](../chapter_18_multi_agent/README.md) only when tasks can be assigned independently.

## Chapter structure

| Section | Focus |
|---|---|
| [15.1 What Is DeepSeek Harness: Cordis Kernel and "Everything Is a Plugin"](./01_what_is_dsh.md) | Express plugin goals as component contracts |
| [15.2 Installation & 4 Run Modes](./02_install_and_modes.md) | Verify entry points and modes in the selected version |
| [15.3 Architecture: Cordis Microkernel & Plugin Topology](./03_cordis_architecture.md) | Map services, events, and component dependencies |
| [15.4 Plugin Development: tool / llm / skill / subagent](./04_plugin_development.md) | Define an extension's inputs, outputs, and failure cleanup |
| [15.5 Comparison: DSH vs Claude Code / OpenClaw / Hermes](./05_comparison.md) | Compare by task and permissions; record unknowns |
| [15.6 Borrowing the Philosophy](./06_lessons_philosophy.md) | Separate transferable designs from product-specific implementations |
| [15.7 Summary: Choose an Implementation with Task Evidence](./07_summary_and_decision_matrix.md) | Choose with fixed tasks, failure cases, and stop conditions |

## Reading path

Review [chapter 8, Harness](../chapter_harness/README.md) and [chapter 9, Skills](../chapter_skill/README.md). [Chapter 14, Claude Code](../chapter_claude_code/README.md) is an optional code-operation comparison rather than a prerequisite. Reinforcement learning and installation of other products are not prerequisites either.

## Verification and material boundaries

The overview does not establish that a release supplies every described interface or run mode, and does not recommend a production platform from a launch slogan. Record exact versions and evidence; verify installation commands, components, and functionality against that version.

Similar plugin interfaces do not establish interchangeable kernels or direct product-to-product UI integration. Comparisons with OpenClaw, Hermes, or Claude Code must keep tasks and permissions consistent; mark unexecuted and unmatched experiments “not run” and “not directly comparable,” respectively.

---

*Next section: [15.1 What Is DeepSeek Harness: Cordis Kernel and "Everything Is a Plugin"](./01_what_is_dsh.md)*
