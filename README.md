<div align="center">

<img src="src/en/svg/preface_learning_path.svg" width="880" alt="The application route and optional research branches">

# Learn AI Agents from Scratch

**A visual, bilingual, and engineering-first textbook for building reliable LLM Agents.**

From Function Calling, memory, planning, RAG, and context engineering to implementation choices, evaluation, security, deployment and capstone delivery, followed by optional research.

<p>
<a href="https://Haozhe-Xing.github.io/agent_learning/en/"><img src="https://img.shields.io/badge/Read%20Online-English-2ea44f?style=for-the-badge" alt="Read the English book"></a>
<a href="https://Haozhe-Xing.github.io/agent_learning/zh/"><img src="https://img.shields.io/badge/在线阅读-中文-d73a49?style=for-the-badge" alt="阅读中文版"></a>
</p>

<p>
<a href="https://github.com/Haozhe-Xing/agent_learning/stargazers"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=flat-square&logo=github&color=gold" alt="GitHub stars"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License"></a>
<a href="https://github.com/Haozhe-Xing/agent_learning/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome"></a>
<img src="https://img.shields.io/badge/Chapters-25-6f42c1?style=flat-square" alt="25 chapters">
<img src="https://img.shields.io/badge/Languages-English%20%7C%20中文-0969da?style=flat-square" alt="Bilingual">
</p>

[中文说明](README_ZH.md) · [Complete directory](#complete-directory) · [Runnable reference agent](#runnable-reference-agent) · [Contributing](#contributing)

</div>

---

## What is this repository?

`agent_learning` is an open-source AI Agent textbook and learning repository. It is designed for the gap between **"I can call an LLM API"** and **"I can build, evaluate, secure, and deploy an Agent system."**

Instead of teaching isolated framework APIs, the book builds one connected mental model:

> **task contract → model baseline → tools → memory → planning → evidence → context → host controls → evaluation → security → deployment → one capstone**

The repository includes:

- **25 chapters across seven parts**: foundations, core capabilities, implementation choices, collaboration/integration, validation/deployment, capstones, and optional research.
- **Aligned English and Chinese navigation**, with a shared project and chapter acceptance handoffs.
- **330+ original SVG diagrams** and **5 interactive demos** for architecture, state, sequence, and training flows.
- **Paper-to-practice explanations** covering ReAct, Reflexion, MemGPT/Letta, GraphRAG, GRPO, MCP, A2A, and more.
- **`reference-agent/`**, a small runnable Agent baseline with tools, memory, security gates, evaluation, an MCP server, a FastAPI service, and offline regression tests.

> This is not an awesome-list and not a framework manual. It is a structured path from first principles to production engineering.

---

## Complete directory

Follow [the continuing project](src/en/learning_project.md). Every chapter states the previous failure, its addition, a verification artifact and the next decision. Frameworks and collaboration are chosen by need; training is in the final optional part. Chapter URLs and example directories retain stable names even where reader-facing numbers have changed.

- **[Part I: Foundations](src/en/part1.md)**
  - [Chapter 1: What Is an Agent?](src/en/chapter_intro/README.md)
  - [Chapter 2: Large Language Model Fundamentals](src/en/chapter_llm/README.md)
- **[Part II: Core Capabilities](src/en/part2.md)**
  - [Chapter 3: Tool Use / Function Calling](src/en/chapter_tools/README.md)
  - [Chapter 4: Memory Systems](src/en/chapter_memory/README.md)
  - [Chapter 5: Planning and Reasoning](src/en/chapter_planning/README.md)
  - [Chapter 6: Retrieval-Augmented Generation (RAG)](src/en/chapter_rag/README.md)
  - [Chapter 7: Context Engineering](src/en/chapter_context_engineering/README.md)
  - [Chapter 8: Harness Engineering: System Engineering for Controlling Agents](src/en/chapter_harness/README.md)
  - [Chapter 9: Skill System](src/en/chapter_skill/README.md)
- **[Part III: Frameworks and Implementation Choices](src/en/part3.md)**
  - [Chapter 10: LangChain In-Depth](src/en/chapter_langchain/README.md)
  - [Chapter 11: LangGraph: Building Stateful Agents](src/en/chapter_langgraph/README.md)
  - [Chapter 12: OpenClaw — Cross-Platform Personal AI Assistant](src/en/chapter_openclaw/README.md)
  - [Chapter 13: Hermes Agent — The Self-Evolving Agent](src/en/chapter_hermes/README.md)
  - [Chapter 14: Deep Dive into Claude Code](src/en/chapter_claude_code/README.md)
  - [Chapter 15: DeepSeek Harness — The Everything-Is-a-Plugin Foundation](src/en/chapter_deepseek_harness/README.md)
- **[Part IV: Collaboration and Integration](src/en/part4.md)**
  - [Chapter 16: Multi-Agent Collaboration](src/en/chapter_18_multi_agent/README.md)
  - [Chapter 17: Agent Communication Protocols](src/en/chapter_19_protocol/README.md)
- **[Part V: Validation and Deployment](src/en/part5.md)**
  - [Chapter 18: Agent Evaluation and Optimization](src/en/chapter_20_evaluation/README.md)
  - [Chapter 19: Security and Reliability](src/en/chapter_21_security/README.md)
  - [Chapter 20: Deployment and Productionization](src/en/chapter_22_deployment/README.md)
- **[Part VI: Capstone Projects](src/en/part6.md)**
  - [Chapter 21: Capstone Project: AI Coding Assistant](src/en/chapter_23_coding_agent/README.md)
  - [Chapter 22: Capstone Project: Intelligent Data Analysis Agent](src/en/chapter_24_data_agent/README.md)
  - [Chapter 23: Capstone Project: Multimodal Agent](src/en/chapter_25_multimodal/README.md)
- **[Part VII: Research Extensions](src/en/part7.md)**
  - [Chapter 24: Agentic-RL: Reinforcement Learning for Agents](src/en/chapter_agentic_rl/README.md)
  - [Chapter 25 Self-Evolution and Continuous Learning](src/en/chapter_self_evolving/README.md)
  - [Appendix F: Development Environment Setup](src/en/chapter_setup/README.md)

[English contents](src/en/SUMMARY.md) · [Chinese contents](src/zh/SUMMARY.md) · [Reference project](reference_code/evidence_assistant/README.md) · [Chapter examples](examples/README.md)

---

## Standalone reference project

The [local evidence research assistant](reference_code/evidence_assistant/README.md) keeps its code, sample documents, tests, and contracts in one directory. It is a small single-agent application: ask several questions about local documents, retrieve evidence, respect execution and context budgets, and produce a cited report. It runs offline with the Python standard library. Framework adapters and external model integration are optional extensions.

```bash
python -m reference_code.evidence_assistant
```

The smaller distance-calculation fixture remains available through `python -m reference_code.evidence_assistant.demo --all` for learning individual mechanisms.

## Runnable reference agent

[`reference-agent/`](reference-agent/) is the shared, dependency-light implementation behind the hands-on chapters. It includes:

- a minimal ReAct loop and tool registry;
- offline `FakeProvider` and optional OpenAI provider;
- memory, prompt-injection guardrails, and fail-closed permission checks;
- an MCP server, FastAPI endpoints, streaming, evaluation harness, and Dockerfile;
- Offline regression tests that run without an API key.

```bash
cd reference-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The implementation is intentionally small enough to read. It is a teaching baseline, not a claim of production completeness.

---

## WeChat

<div align="center">

Feel free to contact the author on WeChat.

<img src="https://assets.with.tencent.com/default/f497918d-1726-441b-9cf2-e7931218bfaa/image-019fe985c3d974bab5dec953f66565d0-01a08b38-41e7-7d63-a565-7b31fdebfc52.png" width="220" alt="WeChat QR code">

</div>

---

## Project principles

1. **Mechanisms before frameworks.** Explain why an abstraction exists before teaching its API.
2. **Visuals must teach.** Diagrams carry architecture and process information; they are not decoration.
3. **Research must lead to engineering insight.** Paper notes include contribution, mechanism, use, and limitations.
4. **Production claims must be honest.** Runnable code, tests, security boundaries, and known limitations are stated explicitly.
5. **Bilingual content stays aligned.** Text, diagrams, navigation, and interactive demos are maintained in both languages.

---

## Contributing

Corrections, clearer explanations, runnable examples, translation fixes, and new paper notes are welcome.

- Found an error? [Open an issue](https://github.com/Haozhe-Xing/agent_learning/issues/new).
- Want to improve a chapter? Edit the matching file under both `src/en/` and `src/zh/` when possible.
- Adding a page? Update both `SUMMARY.md` files.
- Adding a diagram? Place localized assets under `src/en/svg/` and `src/zh/svg/`.
- Before a PR, run `./serve.sh` and verify both language builds.

Please keep claims verifiable and prefer primary sources for papers, protocols, versions, and external projects.

---

## Roadmap

- [x] 25-chapter bilingual mdBook
- [x] Localized diagrams and interactive demos
- [x] Agentic RL, context engineering, harness engineering, and self-evolving Agent coverage
- [x] Runnable `reference-agent` baseline with offline tests
- [ ] More end-to-end capstone implementations
- [ ] Searchable diagram gallery and concept index
- [ ] Evaluation and observability starter templates
- [ ] More exercises, interview questions, and regression cases

Suggestions are welcome in [Issues](https://github.com/Haozhe-Xing/agent_learning/issues).

---

## License

Released under the [MIT License](LICENSE).

<div align="center">

### If this repository saves you time, consider giving it a Star.

A Star helps more engineers find a structured path through AI Agents instead of another disconnected list of links.

[Read in English](https://Haozhe-Xing.github.io/agent_learning/en/) · [阅读中文版](https://Haozhe-Xing.github.io/agent_learning/zh/) · [Open an issue](https://github.com/Haozhe-Xing/agent_learning/issues)

</div>
