# Chapter 14: Deep Dive into Claude Code

**Reading level**: Optional coding-product and architecture case, selected by task.

Use Claude Code as a coding-assistant case to investigate how file operations, command execution, context, and permissions affect a code task. The architecture discussion is a teaching perspective, separating observable behavior, versioned implementation material, and inference.

## This chapter's iteration

- **Previous limitation**: Tools and Harness controls have not yet been checked through code changes: does the patch meet requirements, do tests run, and does failure stop delivery?
- **This chapter adds**: An observation method for code operations, context, permissions, and extension mechanisms in a coding-product case.
- **How to verify**: No uniform product adapter is supplied. Fix a repository, task, version, and permissions; independently inspect patches and test outcomes and keep authorization and failure records. Design review without execution must be marked “not run.”
- **Next step**: For component replacement, select [chapter 15, DeepSeek Harness](../chapter_deepseek_harness/README.md). To implement a coding assistant, continue to the [chapter 21 capstone](../chapter_23_coding_agent/README.md).

## Chapter structure

| Section | Focus |
|---|---|
| [14.1 The Prehistory of Industrial Harnesses: From AutoGPT to Claude Code](./01_industry_history.md) | Understand the task background of coding and execution loops |
| [14.2 Getting to Know Claude Code](./02_introduction.md) | Record interaction mode and experiment configuration |
| [14.3 Deep Dive into Core Architecture](./03_architecture.md) | Use the book's analysis model to map tools and state |
| [14.4 System Prompt, Permission Engineering & Prompt Cache](./04_system_prompt_and_permissions.md) | Prompt, permission, context, and cache boundaries |
| [14.5 Advanced Usage: MCP, Hooks, and Skills](./05_advanced_usage.md) | Inspect authorization and outcomes at extension interfaces |
| [14.6 Production Practice](./06_production_and_team.md) | Team configuration, failures, and acceptance requirements |

## Reading path

The relevant foundations are [chapter 3, Tools](../chapter_tools/README.md), [chapter 7, Context](../chapter_context_engineering/README.md), and [chapter 8, Harness](../chapter_harness/README.md). Product cases in chapters 12–13 are not prerequisites. Start at 14.2 for use, or select 14.3–14.4 for architectural study.

## Verification and material boundaries

Selection does not rely on licensing claims, line-by-line access to internal source, fixed prompt lengths, or tool counts. If a structural claim comes from community analysis, identify the material and version and label inference. Observable behavior still needs reproduction in the reader's experiment environment.

The repository's development-team and coding-foundation examples verify their own code paths, not Claude Code internals or live-model quality. Use independent acceptance for generated code; a product's completion message does not establish successful delivery.

---

*Next section: [14.1 The Prehistory of Industrial Harnesses: From AutoGPT to Claude Code](./01_industry_history.md)*
