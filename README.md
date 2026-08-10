<div align="center">

<img src="readme_img.png" width="880" alt="Agent Learning Roadmap">

# Learn AI Agents from Scratch

**A visual, bilingual, and engineering-first textbook for building reliable LLM Agents.**

From Function Calling, memory, planning, RAG, and context engineering to Agentic RL, multi-agent systems, evaluation, security, and production deployment.

<p>
<a href="https://Haozhe-Xing.github.io/agent_learning/en/"><img src="https://img.shields.io/badge/Read%20Online-English-2ea44f?style=for-the-badge" alt="Read the English book"></a>
<a href="https://Haozhe-Xing.github.io/agent_learning/zh/"><img src="https://img.shields.io/badge/在线阅读-中文-d73a49?style=for-the-badge" alt="阅读中文版"></a>
</p>

<p>
<a href="https://github.com/Haozhe-Xing/agent_learning/stargazers"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=flat-square&logo=github&color=gold" alt="GitHub stars"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="MIT License"></a>
<a href="https://github.com/Haozhe-Xing/agent_learning/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?style=flat-square" alt="PRs welcome"></a>
<img src="https://img.shields.io/badge/Chapters-23-6f42c1?style=flat-square" alt="23 chapters">
<img src="https://img.shields.io/badge/Languages-English%20%7C%20中文-0969da?style=flat-square" alt="Bilingual">
</p>

[中文说明](README_ZH.md) · [Complete directory](#complete-directory) · [Runnable reference agent](#runnable-reference-agent) · [Contributing](#contributing)

</div>

---

## What is this repository?

`agent_learning` is an open-source AI Agent textbook and learning repository. It is designed for the gap between **"I can call an LLM API"** and **"I can build, evaluate, secure, and deploy an Agent system."**

Instead of teaching isolated framework APIs, the book builds one connected mental model:

> **LLM fundamentals → tools → memory → planning → RAG → context → harness → skills → Agentic RL → multi-agent → evaluation → security → deployment**

The repository includes:

- **23 chapters** across foundations, core capabilities, frameworks, multi-agent systems, production, and capstone projects.
- **188 Markdown pages per language**, maintained in English and Chinese.
- **330+ original SVG diagrams** and **5 interactive demos** for architecture, state, sequence, and training flows.
- **Paper-to-practice explanations** covering ReAct, Reflexion, MemGPT/Letta, GraphRAG, GRPO, MCP, A2A, and more.
- **`reference-agent/`**, a small runnable Agent baseline with tools, memory, security gates, evaluation, an MCP server, a FastAPI service, and 16 tests.

> This is not an awesome-list and not a framework manual. It is a structured path from first principles to production engineering.

---

## Complete directory

- **Book source**
  - `src/en/` — English mdBook source
  - `src/zh/` — Chinese mdBook source
  - `src/en/SUMMARY.md` — English table of contents
  - `src/zh/SUMMARY.md` — Chinese table of contents
- **Foundations**
  - [1. What Is an Agent?](https://Haozhe-Xing.github.io/agent_learning/en/chapter_intro/)
  - [2. LLM Fundamentals](https://Haozhe-Xing.github.io/agent_learning/en/chapter_llm/)
- **Core capabilities**
  - [3. Tools](https://Haozhe-Xing.github.io/agent_learning/en/chapter_tools/)
  - [4. Memory](https://Haozhe-Xing.github.io/agent_learning/en/chapter_memory/)
  - [5. Planning](https://Haozhe-Xing.github.io/agent_learning/en/chapter_planning/)
  - [6. RAG](https://Haozhe-Xing.github.io/agent_learning/en/chapter_rag/)
  - [7. Context Engineering](https://Haozhe-Xing.github.io/agent_learning/en/chapter_context_engineering/)
  - [8. Harness Engineering](https://Haozhe-Xing.github.io/agent_learning/en/chapter_harness/)
  - [9. Skills](https://Haozhe-Xing.github.io/agent_learning/en/chapter_skill/)
  - [10. Agentic RL](https://Haozhe-Xing.github.io/agent_learning/en/chapter_agentic_rl/)
  - [11. Self-Evolving Agents](https://Haozhe-Xing.github.io/agent_learning/en/chapter_self_evolving/)
- **Framework practice**
  - [12. LangChain](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langchain/)
  - [13. LangGraph](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langgraph/)
  - [14. Agent Frameworks](https://Haozhe-Xing.github.io/agent_learning/en/chapter_frameworks/)
  - [15. Claude Code](https://Haozhe-Xing.github.io/agent_learning/en/chapter_claude_code/)
- **Multi-agent systems**
  - [16. Multi-Agent Collaboration](https://Haozhe-Xing.github.io/agent_learning/en/chapter_multi_agent/)
  - [17. Agent Protocols](https://Haozhe-Xing.github.io/agent_learning/en/chapter_protocol/)
- **Production engineering**
  - [18. Evaluation](https://Haozhe-Xing.github.io/agent_learning/en/chapter_evaluation/)
  - [19. Security](https://Haozhe-Xing.github.io/agent_learning/en/chapter_security/)
  - [20. Deployment](https://Haozhe-Xing.github.io/agent_learning/en/chapter_deployment/)
- **Capstone projects**
  - [21. Coding Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_coding_agent/)
  - [22. Data Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_data_agent/)
  - [23. Multimodal Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_multimodal/)
- **Appendices**
  - [Prompt templates](https://Haozhe-Xing.github.io/agent_learning/en/appendix/prompt_templates.html)
  - [FAQ](https://Haozhe-Xing.github.io/agent_learning/en/appendix/faq.html)
  - [Resources](https://Haozhe-Xing.github.io/agent_learning/en/appendix/resources.html)
  - [Glossary](https://Haozhe-Xing.github.io/agent_learning/en/appendix/glossary.html)
  - [KL divergence](https://Haozhe-Xing.github.io/agent_learning/en/appendix/kl_divergence.html)
  - [Environment setup](https://Haozhe-Xing.github.io/agent_learning/en/chapter_setup/)
- **Runnable implementation**
  - `reference-agent/` — teaching baseline Agent implementation
  - `reference-agent/src/reference_agent/` — Agent loop, providers, tools, memory, security, server, evaluation
  - `reference-agent/tests/` — offline test suite
- **Assets and build files**
  - `src/en/svg/` — English diagrams
  - `src/zh/svg/` — Chinese diagrams
  - `src/en/animations/` — English interactive demos
  - `src/zh/animations/` — Chinese interactive demos
  - `theme/` — shared mdBook theme
  - `book.toml` — Chinese mdBook config
  - `book-en.toml` — English mdBook config
  - `serve.sh` — build and serve both books locally

---

## Runnable reference agent

[`reference-agent/`](reference-agent/) is the shared, dependency-light implementation behind the hands-on chapters. It includes:

- a minimal ReAct loop and tool registry;
- offline `FakeProvider` and optional OpenAI provider;
- memory, prompt-injection guardrails, and fail-closed permission checks;
- an MCP server, FastAPI endpoints, streaming, evaluation harness, and Dockerfile;
- **16 tests** that run without an API key.

```bash
cd reference-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The implementation is intentionally small enough to read. It is a teaching baseline, not a claim of production completeness.


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

- [x] 23-chapter bilingual mdBook
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
