# Chapter 21: Project Practice: AI Coding Assistant

**Reading level**: Capstone; choose at least one domain from chapters 21–23.

## This chapter's iteration

- **Previous limitation**: Passing a generic numeric task does not establish repository diagnosis, bounded edits or an independently tested patch.
- **This chapter adds**: Replace document evidence with code indexes and restrict actions to reviewable changes; define acceptance for understanding, generation, testing and rollback.
- **How to verify**: This is a component-design and assembly exercise, not a supplied complete coding assistant. Repair one small defect and retain the original failure, passing result, unchanged independent tests and diff. Out-of-scope edits, deleted tests or failed rollback cannot count as completion.
- **Next step**: Finish the application route after acceptance. [Data analysis](../chapter_24_data_agent/README.md) and [multimodal tasks](../chapter_25_multimodal/README.md) are alternative capstones.

> Build an AI coding assistant from scratch that can understand code, generate code, write tests, and fix bugs.

---

## Chapter Overview

This chapter is the first comprehensive hands-on project. We will apply the core Agent technologies learned so far — tool calling, memory systems, RAG, planning and reasoning — to build an AI coding assistant that truly helps developers.

## Chapter Goals

- ✅ Design the overall architecture of a coding assistant
- ✅ Implement code indexing and semantic search
- ✅ Build code generation and modification capabilities
- ✅ Implement automatic test generation and bug fixing
- ✅ Integrate everything into a complete, usable tool

## Chapter Structure

| Section |
|---|
| [21.1 Project Architecture Design](./01_architecture.md) |
| [21.2 Code Understanding and Analysis](./02_code_understanding.md) |
| [21.3 Code Generation and Modification](./03_code_generation.md) |
| [21.4 Test Generation and Bug Fixing](./04_testing_debugging.md) |
| [21.5 Integration Design: From Components to Assistant](./05_full_implementation.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Finish the application route after acceptance. [Data analysis](../chapter_24_data_agent/README.md) and [multimodal tasks](../chapter_25_multimodal/README.md) are alternative capstones.

---

*Next: [21.1 Project Architecture Design](./01_architecture.md)*
