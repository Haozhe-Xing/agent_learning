<div align="center">

<img src="readme_img.png" width="880" alt="Agent Learning">

<h1>Learn AI Agents from Scratch</h1>

<p><b>The missing textbook between "I can call an LLM API" and "I ship Agents to production."</b></p>

<p>
23 chapters · <b>336 original diagrams</b> · 5 interactive animations · runnable reference implementation<br>
Bilingual (EN / 中文) · <b>actively tracks new arXiv Agent papers</b>
</p>

<p>
<a href="https://Haozhe-Xing.github.io/agent_learning/en/"><img src="https://img.shields.io/badge/📖_Read_the_Book-4CAF50?style=for-the-badge" alt="Read online"></a>
&nbsp;
<a href="https://Haozhe-Xing.github.io/agent_learning/en/chapter_setup/04_hello_agent.html"><img src="https://img.shields.io/badge/⚡_First_Agent_in_30min-FF6F00?style=for-the-badge" alt="Quickstart"></a>
&nbsp;
<a href="#-full-curriculum"><img src="https://img.shields.io/badge/🗺️_Curriculum-2196F3?style=for-the-badge" alt="Curriculum"></a>
</p>

<p>
<a href="https://github.com/Haozhe-Xing/agent_learning/stargazers"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=flat-square&logo=github&color=FFD700" alt="Stars"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT"></a>
<a href="https://github.com/Haozhe-Xing/agent_learning/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
<img src="https://img.shields.io/badge/arXiv-tracked_weekly-red?style=flat-square&logo=arxiv" alt="arXiv tracking">
<img src="https://img.shields.io/badge/lang-EN_|_中文-blue?style=flat-square" alt="Bilingual">
</p>

<a href="README_ZH.md">🇨🇳 中文版 README</a> · <a href="https://Haozhe-Xing.github.io/agent_learning/zh/">🇨🇳 在线阅读</a>

</div>

---

## Why this repo exists

Most Agent learning resources fall into one of three traps:

|  | The problem |
| --- | --- |
| **Awesome-lists** | 500 links, zero structure. You don't know what to read first or how pieces connect. |
| **Framework docs** | Teach you *LangGraph's API*, not *why stateful graphs beat while-loops*. Knowledge dies with the framework version. |
| **Blog tutorials** | "Build a RAG chatbot in 10 minutes" — then silence on evaluation, cost, prompt injection, context rot, or why it breaks at step 12. |

**This book is the connective tissue.** It builds one coherent mental model — tools → memory → planning → context → harness → skills → RL → multi-agent → production — where every chapter explains *the problem that forced this technique to exist* before showing code.

<table>
<tr><td width="33%" valign="top">

### 📐 Diagrams that carry the load
**336 hand-made SVGs.** Not decoration — architecture diagrams, sequence flows, and state machines that make abstract loops concrete. Most are the fastest path to understanding in their chapter.

</td><td width="33%" valign="top">

### 🔬 Papers, digested
Every core chapter ends with paper deep-dives: the problem at the time, the mechanism, the engineering takeaway, **and the limitation**. ReAct, Reflexion, MemGPT, GRPO, STaR/V-STaR, HippoRAG, and 100+ more.

</td><td width="33%" valign="top">

### 📡 Never goes stale
New Agent research from arXiv gets digested and folded into the matching chapters on a rolling basis — typically every few days. The frontier sections you read are **weeks old, not years**.

</td></tr>
</table>

---

## 🎯 Start where you are

| If you… | Read this path | You'll walk away able to |
| --- | --- | --- |
| **have never built an Agent** | [Ch.1 What is an Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_intro/) → [Ch.2 LLM Basics](https://Haozhe-Xing.github.io/agent_learning/en/chapter_llm/) → [Hello Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_setup/04_hello_agent.html) | Explain the perceive-think-act loop and run your own tool-using Agent |
| **can prompt, want to build** | [Ch.3 Tools](https://Haozhe-Xing.github.io/agent_learning/en/chapter_tools/) → [Ch.4 Memory](https://Haozhe-Xing.github.io/agent_learning/en/chapter_memory/) → [Ch.5 Planning](https://Haozhe-Xing.github.io/agent_learning/en/chapter_planning/) → [Ch.6 RAG](https://Haozhe-Xing.github.io/agent_learning/en/chapter_rag/) | Assemble a real application from tool calling, retrieval, memory, and planning |
| **have a demo that breaks in prod** | [Ch.7 Context Engineering](https://Haozhe-Xing.github.io/agent_learning/en/chapter_context_engineering/) → [Ch.8 Harness](https://Haozhe-Xing.github.io/agent_learning/en/chapter_harness/) → [Ch.18 Evaluation](https://Haozhe-Xing.github.io/agent_learning/en/chapter_evaluation/) → [Ch.19 Security](https://Haozhe-Xing.github.io/agent_learning/en/chapter_security/) | Diagnose context rot, add permission gates, build an eval suite, defend against injection |
| **want to train Agents, not just prompt them** | [Ch.10 Agentic-RL](https://Haozhe-Xing.github.io/agent_learning/en/chapter_agentic_rl/) → [Ch.11 Self-Evolving](https://Haozhe-Xing.github.io/agent_learning/en/chapter_self_evolving/) | Understand SFT/LoRA, PPO vs DPO vs GRPO, and how data flywheels compound |
| **need to pick a framework** | [Ch.12 LangChain](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langchain/) → [Ch.13 LangGraph](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langgraph/) → [Ch.14 Framework Landscape](https://Haozhe-Xing.github.io/agent_learning/en/chapter_frameworks/) | Choose deliberately instead of by GitHub star count |

> 💡 **Don't read linearly.** The [learning-path overview](https://Haozhe-Xing.github.io/agent_learning/en/chapter_intro/) maps three curated routes (5-chapter fast track / 11-chapter engineer / 8-chapter researcher) through the book.

---

## 👀 What it actually looks like

> A sample of the **336 original SVGs** — every diagram in this book was drawn for it.

<table>
<tr>
<td width="50%" align="center">
<b>Perceive → Think → Act</b> · Ch.1<br>
<img src="src/en/svg/chapter_intro_03_loop.svg" width="410" alt="Perceive-Think-Act Loop"><br>
<sub>The loop that separates an Agent from a chatbot</sub>
</td>
<td width="50%" align="center">
<b>ReAct reasoning</b> · Ch.5<br>
<img src="src/en/svg/chapter_planning_02_react_loop.svg" width="410" alt="ReAct Framework"><br>
<sub>Thought → Action → Observation, interleaved</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>Function Calling, end to end</b> · Ch.3<br>
<img src="src/en/svg/chapter_tools_02_function_calling.svg" width="470" alt="Function Calling Flow"><br>
<sub>All 6 hops, including the message shapes people get wrong</sub>
</td>
<td width="50%" align="center">
<b>RAG: offline index + online retrieve</b> · Ch.6<br>
<img src="src/en/svg/chapter_rag_01_rag_flow.svg" width="470" alt="RAG Workflow"><br>
<sub>Where grounding actually happens</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>Three-layer memory</b> · Ch.4<br>
<img src="src/en/svg/chapter_memory_01_memory_types.svg" width="470" alt="Memory Architecture"><br>
<sub>What sinks to long-term, what gets pulled back up</sub>
</td>
<td width="50%" align="center">
<b>Prompt vs Context Engineering</b> · Ch.7<br>
<img src="src/en/svg/chapter_context_01_comparison.svg" width="470" alt="Context Engineering"><br>
<sub>From "how to phrase it" to "what the model sees"</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>MCP / A2A / ANP protocol stack</b> · Ch.17<br>
<img src="src/en/svg/chapter_protocol_03_three_protocols.svg" width="470" alt="Protocol Comparison"><br>
<sub>Discovery → task collaboration → tool invocation</sub>
</td>
<td width="50%" align="center">
<b>GRPO training architecture</b> · Ch.10<br>
<img src="src/en/svg/chapter_agentic_rl_03_grpo_architecture.svg" width="410" alt="GRPO Architecture"><br>
<sub>No critic model — advantage via in-group normalization</sub>
</td>
</tr>
</table>

### Plus 5 interactive animations

| Animation | Chapter | What it makes clickable |
| --- | --- | --- |
| 🔄 Perceive-Think-Act cycle | Ch.1 | Step through a full Agent turn |
| 💡 ReAct reasoning | Ch.5 | Watch thought and action interleave |
| 🔧 Function Calling | Ch.3 | Trace the tool round-trip |
| 📚 RAG pipeline | Ch.6 | Chunk → embed → retrieve → generate |
| 🎯 GRPO sampling | Ch.10 | In-group sampling and reward normalization |

<div align="center"><sub>Animations run in the <a href="https://Haozhe-Xing.github.io/agent_learning/en/">online book</a> (and any local build).</sub></div>

---

## 📚 Full curriculum

<details open>
<summary><b>Part I — Getting Started</b> · build correct intuitions before writing code</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 1 | [What Is an Agent?](https://Haozhe-Xing.github.io/agent_learning/en/chapter_intro/) | Agent vs chatbot vs traditional program · perceive-think-act loop · history from symbolic AI to LLMs |
| 2 | [LLM Fundamentals](https://Haozhe-Xing.github.io/agent_learning/en/chapter_llm/) | Tokenizer/BPE · attention derivation · KV cache · RoPE · prompting strategies · model selection |

</details>

<details open>
<summary><b>Part II — Core Capabilities</b> · the eight abilities every serious Agent needs</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 3 | [Tool Use / Function Calling](https://Haozhe-Xing.github.io/agent_learning/en/chapter_tools/) | The calling mechanism · custom tools · writing descriptions models actually follow |
| 4 | [Memory Systems](https://Haozhe-Xing.github.io/agent_learning/en/chapter_memory/) | Short/long/working memory · vector retrieval · MemGPT & Letta architecture |
| 5 | [Planning & Reasoning](https://Haozhe-Xing.github.io/agent_learning/en/chapter_planning/) | ReAct · task decomposition · reflection · Plan-and-Execute · test-time compute scaling |
| 6 | [RAG](https://Haozhe-Xing.github.io/agent_learning/en/chapter_rag/) | Chunking · embeddings · reranking · **GraphRAG & Agentic RAG** |
| 7 | [Context Engineering](https://Haozhe-Xing.github.io/agent_learning/en/chapter_context_engineering/) | Attention budget · context rot · long-horizon strategies · build a context manager |
| 8 | [Harness Engineering](https://Haozhe-Xing.github.io/agent_learning/en/chapter_harness/) | Six engineering pillars · `AGENTS.md` / `CLAUDE.md` · reliable structured output |
| 9 | [Skill System](https://Haozhe-Xing.github.io/agent_learning/en/chapter_skill/) | Skill definition & discovery · tool vs skill vs sub-agent · `SKILL.md` ecosystem |
| 10 | [Agentic-RL](https://Haozhe-Xing.github.io/agent_learning/en/chapter_agentic_rl/) | SFT+LoRA · DP/TP/PP/ZeRO · **PPO vs DPO vs GRPO** · fine-tuning for Agents |

</details>

<details>
<summary><b>Part III — Frameworks & Self-Evolution</b> · pick tools deliberately</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 11 | [Self-Evolving Agents](https://Haozhe-Xing.github.io/agent_learning/en/chapter_self_evolving/) | Automatic prompt optimization (APE/OPRO/DSPy/GEPA) · **Agentic data flywheel** |
| 12 | [LangChain In-Depth](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langchain/) | Beyond the quickstart — where the abstractions help and hurt |
| 13 | [LangGraph](https://Haozhe-Xing.github.io/agent_learning/en/chapter_langgraph/) | State/node/edge · why FSMs beat while-loops for real Agents |
| 14 | [Framework Landscape](https://Haozhe-Xing.github.io/agent_learning/en/chapter_frameworks/) | CrewAI · AutoGen · Semantic Kernel · how to actually choose |
| 15 | [Claude Code Deep Dive](https://Haozhe-Xing.github.io/agent_learning/en/chapter_claude_code/) | From usage to internals · best practices for terminal Agents |

</details>

<details>
<summary><b>Part IV — Multi-Agent Systems</b> · when one Agent isn't enough</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 16 | [Multi-Agent Collaboration](https://Haozhe-Xing.github.io/agent_learning/en/chapter_multi_agent/) | Supervisor vs decentralized · message queue / blackboard / direct call |
| 17 | [Communication Protocols](https://Haozhe-Xing.github.io/agent_learning/en/chapter_protocol/) | **MCP · A2A · ANP** — the emerging three-layer protocol stack |

</details>

<details>
<summary><b>Part V — Production</b> · the part tutorials skip</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 18 | [Evaluation & Optimization](https://Haozhe-Xing.github.io/agent_learning/en/chapter_evaluation/) | GAIA · SWE-bench · LLM-as-judge calibration · observability · cost |
| 19 | [Security & Reliability](https://Haozhe-Xing.github.io/agent_learning/en/chapter_security/) | **Prompt injection** (direct & indirect) · permission sandboxing · fail-closed design |
| 20 | [Deployment](https://Haozhe-Xing.github.io/agent_learning/en/chapter_deployment/) | API services · streaming · containers · scaling |

</details>

<details>
<summary><b>Part VI — Capstone Projects</b> + appendices</summary>

| Ch | Title | Highlights |
| --- | --- | --- |
| 21 | [AI Coding Assistant](https://Haozhe-Xing.github.io/agent_learning/en/chapter_coding_agent/) | Read a repo, edit files, run tests, self-repair loop |
| 22 | [Data Analysis Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_data_agent/) | Sandboxed code execution · pipeline vs agentic loop |
| 23 | [Multimodal Agent](https://Haozhe-Xing.github.io/agent_learning/en/chapter_multimodal/) | Vision + text tool use |

**Appendices:** [Prompt templates](https://Haozhe-Xing.github.io/agent_learning/en/appendix/prompt_templates.html) · [FAQ](https://Haozhe-Xing.github.io/agent_learning/en/appendix/faq.html) · [Resources](https://Haozhe-Xing.github.io/agent_learning/en/appendix/resources.html) · [Glossary](https://Haozhe-Xing.github.io/agent_learning/en/appendix/glossary.html) · [KL divergence explained](https://Haozhe-Xing.github.io/agent_learning/en/appendix/kl_divergence.html) · [Environment setup](https://Haozhe-Xing.github.io/agent_learning/en/chapter_setup/)

</details>

---

## ⚡ Quick start

**Just want to read?** → [**Open the book**](https://Haozhe-Xing.github.io/agent_learning/en/). Nothing to install.

<details>
<summary><b>Build it locally</b> (mdBook)</summary>

```bash
# 1. Install mdBook + KaTeX plugin
cargo install mdbook mdbook-katex
# macOS alternative: brew install mdbook

# 2. Clone and serve (builds EN + 中文, port 3000)
git clone https://github.com/Haozhe-Xing/agent_learning.git
cd agent_learning
./serve.sh
```

| URL | Content |
| --- | --- |
| `http://localhost:3000` | Language picker |
| `http://localhost:3000/en/` | English book |
| `http://localhost:3000/zh/` | 中文版 |

</details>

<details>
<summary><b>Run the code</b> (Python 3.11+)</summary>

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install langchain langchain-openai langgraph openai anthropic
export OPENAI_API_KEY="sk-..."
```

The repo also ships **`reference-agent/`** — a minimal, dependency-light Agent implementation (loop, tool registry, memory, permission gate) used as the shared baseline across the hands-on chapters. Read it when framework abstractions start feeling like magic.

</details>

<div align="center">

![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-191919?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat)
![mdBook](https://img.shields.io/badge/mdBook-000000?style=flat&logo=rust&logoColor=white)

</div>

---

## ❓ FAQ

<details>
<summary><b>Do I need to know machine learning?</b></summary>

No. Chapters 1–9 require only Python. Math appears in Ch.10 (Agentic-RL) with KaTeX-rendered derivations, and even there the intuition comes before the formulas — plus [Appendix E](https://Haozhe-Xing.github.io/agent_learning/en/appendix/kl_divergence.html) explains KL divergence from scratch. Skip Ch.10–11 entirely if you only want to build.

</details>

<details>
<summary><b>Will this be obsolete when LangChain ships a breaking change?</b></summary>

Mostly no — and that's deliberate. The book teaches *mechanisms* (why memory needs tiers, why context rots, why tool descriptions are prompts) and treats frameworks as implementations of those mechanisms. Framework-specific chapters (12–15) are the ones that need maintenance; the conceptual core doesn't.

</details>

<details>
<summary><b>How is this different from a paper-reading list?</b></summary>

Papers here are always attached to an engineering question. Each write-up states the problem at the time, the mechanism, **the takeaway for your code**, and the limitation — then a comparison table synthesizes how a group of papers relate. The goal is that you can *use* the idea, not just cite it.

</details>

<details>
<summary><b>Is the Chinese version a machine translation?</b></summary>

The book is authored bilingually with both versions maintained in sync (`src/zh/` and `src/en/`), including separate diagram sets. Neither is a raw MT dump of the other.

</details>

<details>
<summary><b>What does "actively tracks arXiv" actually mean?</b></summary>

arXiv is swept for new Agent-related work (architectures, memory, tools, multi-agent, RL for agents, security), and relevant papers are digested into the "latest advances" section of the matching chapter. In practice this lands every few days rather than strictly every 24h — see the [commit history](https://github.com/Haozhe-Xing/agent_learning/commits/main) for the actual cadence. The point is that frontier sections keep growing without waiting for a book revision.

</details>

---

## 🤝 Contributing

Every kind of contribution is welcome — typo fixes count.

| | |
| --- | --- |
| 🐛 Found an error | [Open an issue](https://github.com/Haozhe-Xing/agent_learning/issues) |
| 💡 Chapter is confusing | [Tell us where](https://github.com/Haozhe-Xing/agent_learning/issues/new) — unclear explanations are bugs |
| 📝 Want to improve content | Fork → edit → PR |
| 🌍 Translation fixes | Both `src/zh/` and `src/en/` are open for improvement |

```bash
git clone https://github.com/YOUR_USERNAME/agent_learning.git
git checkout -b feature/improve-chapter-3
./serve.sh                # preview locally
git commit -m "feat: clarify tool description examples in Ch.3"
```

<details>
<summary><b>Repo conventions</b></summary>

- Chapters live in `src/zh/chapter_xxx/` and `src/en/chapter_xxx/`
- Chapter overview → `README.md`; sections numbered `01_xxx.md`, `02_xxx.md`
- Diagrams → `src/{zh,en}/svg/`, named `chapter_<name>_<desc>.svg`
- Animations → `src/{zh,en}/animations/`
- Add new pages to `SUMMARY.md` in **both** languages

</details>

<details>
<summary><b>Paper write-up template</b></summary>

Frontier sections follow one structure so readers can judge relevance fast:

```markdown
### Paper Title: the problem it solves, in one sentence

- **Paper link**:
- **Code / project link**:
- **Year / organization**:
- **Problem at the time**:
- **Core contribution**:
- **Method breakdown**:
- **Engineering insight for Agent systems**:
- **Limitations**:
```

Quality bar: link the primary source · explain the *historical* contribution, not just the abstract · connect it to memory/tools/planning/eval/safety/training · state what it does **not** solve · and after several papers, add a synthesis table instead of leaving a list.

</details>

---

## 🗺️ Roadmap

**Done** — ✅ Bilingual mdBook · ✅ 336 original diagrams · ✅ Interactive animations · ✅ Paper deep-dives across core chapters · ✅ Agentic-RL (PPO/DPO/GRPO) · ✅ `reference-agent/` teaching baseline · ✅ Rolling arXiv paper tracking

**Next** — ⬜ More runnable end-to-end project templates · ⬜ Agent keyword cheat sheet · ⬜ Diagram gallery index · ⬜ Interview questions & self-check exercises · ⬜ Production template with eval + observability wired in

Have a request? [Open an issue](https://github.com/Haozhe-Xing/agent_learning/issues/new) — the roadmap is reader-driven.

---

<div align="center">

### ⭐ Star it if it saved you time

Not for vanity metrics — Stars are how the next engineer stuck on context rot,
prompt injection, or "which framework?" finds this repo instead of a 500-link awesome-list.

<a href="https://github.com/Haozhe-Xing/agent_learning"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=for-the-badge&logo=github&color=FFD700&label=Star%20this%20repo" alt="Star"></a>

[![Star History Chart](https://api.star-history.com/svg?repos=Haozhe-Xing/agent_learning&type=Date)](https://www.star-history.com/#Haozhe-Xing/agent_learning&Date)

<br>

**[📖 Read the book](https://Haozhe-Xing.github.io/agent_learning/en/)** · **[🇨🇳 中文版](https://Haozhe-Xing.github.io/agent_learning/zh/)** · **[🐛 Issues](https://github.com/Haozhe-Xing/agent_learning/issues)** · **[📄 MIT License](LICENSE)**

<sub>Built so that understanding Agents doesn't require reading 200 papers and 12 framework changelogs.</sub>

</div>
