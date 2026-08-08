<div align="center">

<img src="readme_img.png" width="880" alt="Agent Learning">

<h1>从零开始学 AI Agent</h1>

<p><b>一本填补空白的教科书——介于「我会调用 LLM API」和「我能把 Agent 送上生产环境」之间。</b></p>

<p>
23 章 · <b>336 张原创图解</b> · 5 个交互动画 · 可运行的参考实现<br>
中英双语 · <b>持续追踪 arXiv 最新 Agent 论文</b>
</p>

<p>
<a href="https://Haozhe-Xing.github.io/agent_learning/zh/"><img src="https://img.shields.io/badge/📖_开始阅读-4CAF50?style=for-the-badge" alt="在线阅读"></a>
&nbsp;
<a href="https://Haozhe-Xing.github.io/agent_learning/zh/chapter_setup/04_hello_agent.html"><img src="https://img.shields.io/badge/⚡_30分钟跑通第一个Agent-FF6F00?style=for-the-badge" alt="快速开始"></a>
&nbsp;
<a href="#-完整目录"><img src="https://img.shields.io/badge/🗺️_完整目录-2196F3?style=for-the-badge" alt="目录"></a>
</p>

<p>
<a href="https://github.com/Haozhe-Xing/agent_learning/stargazers"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=flat-square&logo=github&color=FFD700" alt="Stars"></a>
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square" alt="MIT"></a>
<a href="https://github.com/Haozhe-Xing/agent_learning/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
<img src="https://img.shields.io/badge/arXiv-持续追踪-red?style=flat-square&logo=arxiv" alt="arXiv tracking">
<img src="https://img.shields.io/badge/语言-中文_|_EN-blue?style=flat-square" alt="双语">
</p>

<a href="README.md">🇺🇸 English README</a> · <a href="https://Haozhe-Xing.github.io/agent_learning/en/">🇺🇸 Read in English</a>

</div>

---

## 为什么会有这个仓库

现在的 Agent 学习资料，大多掉进三个坑里：

|  | 问题在哪 |
| --- | --- |
| **Awesome 列表** | 500 个链接，零结构。你不知道该先读哪个，也不知道这些东西怎么串起来。 |
| **框架文档** | 教你 *LangGraph 的 API*，不教你*为什么有状态图比 while 循环更适合 Agent*。知识随版本号一起过期。 |
| **博客教程** | 「10 分钟搭一个 RAG 机器人」——然后对评估、成本、提示注入、上下文腐化、以及它为什么在第 12 步崩掉，集体沉默。 |

**这本书是把它们连起来的结缔组织。** 它构建一条连贯的认知主线——工具 → 记忆 → 规划 → 上下文 → Harness → 技能 → 强化学习 → 多 Agent → 生产化，而且每一章都先讲清楚*是什么问题逼出了这项技术*，再给代码。

<table>
<tr><td width="33%" valign="top">

### 📐 图解承担讲解主力
**336 张手工绘制的 SVG。** 不是装饰——架构图、时序流程、状态机，把抽象的循环变具体。很多章节里，看图是理解最快的路径。

</td><td width="33%" valign="top">

### 🔬 论文，嚼碎了喂
每个核心章节都以论文精读收尾：当时要解决什么问题、机制是什么、对工程的启示，**以及局限在哪**。ReAct、Reflexion、MemGPT、GRPO、STaR/V-STaR、HippoRAG，以及 100+ 篇。

</td><td width="33%" valign="top">

### 📡 内容不会过期
arXiv 上的新 Agent 研究会被持续消化、并入对应章节——通常每隔几天一批。你读到的前沿部分，**是几周前的，不是几年前的**。

</td></tr>
</table>

---

## 🎯 从你现在的位置开始

| 如果你… | 走这条路线 | 读完你能 |
| --- | --- | --- |
| **从没写过 Agent** | [第1章 什么是 Agent](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_intro/) → [第2章 大模型基础](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_llm/) → [Hello Agent](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_setup/04_hello_agent.html) | 讲清感知-思考-行动循环，并跑通自己的第一个会用工具的 Agent |
| **会写提示词，想做应用** | [第3章 工具](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_tools/) → [第4章 记忆](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_memory/) → [第5章 规划](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_planning/) → [第6章 RAG](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_rag/) | 把工具调用、检索、记忆、规划组装成一个真实应用 |
| **Demo 能跑，上线就崩** | [第7章 上下文工程](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_context_engineering/) → [第8章 Harness](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_harness/) → [第18章 评估](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_evaluation/) → [第19章 安全](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_security/) | 诊断上下文腐化、加权限闸门、搭评估集、防住提示注入 |
| **想训练 Agent，不只调提示词** | [第10章 Agentic-RL](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_agentic_rl/) → [第11章 自我进化](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_self_evolving/) | 搞懂 SFT/LoRA、PPO vs DPO vs GRPO，以及数据飞轮如何复利 |
| **在纠结选哪个框架** | [第12章 LangChain](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_langchain/) → [第13章 LangGraph](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_langgraph/) → [第14章 框架全景](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_frameworks/) | 按需求做选择，而不是按 GitHub Star 数 |

> 💡 **不必线性读。** [学习路径全景图](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_intro/)给出三条精选路线（5 章快速上手 / 11 章工程师 / 8 章研究者）。

---

## 👀 内容长什么样

> 以下是 **336 张原创 SVG** 的抽样——书里每一张图都是为它专门画的。

<table>
<tr>
<td width="50%" align="center">
<b>感知 → 思考 → 行动</b> · 第1章<br>
<img src="src/zh/svg/chapter_intro_03_loop.svg" width="410" alt="感知-思考-行动循环"><br>
<sub>把 Agent 和聊天机器人区分开的那个循环</sub>
</td>
<td width="50%" align="center">
<b>ReAct 推理</b> · 第5章<br>
<img src="src/zh/svg/chapter_planning_02_react_loop.svg" width="410" alt="ReAct 框架"><br>
<sub>思考 → 行动 → 观察，交替推进</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>Function Calling 完整链路</b> · 第3章<br>
<img src="src/zh/svg/chapter_tools_02_function_calling.svg" width="470" alt="Function Calling 流程"><br>
<sub>全部 6 跳，含最容易写错的消息结构</sub>
</td>
<td width="50%" align="center">
<b>RAG：离线索引 + 在线检索</b> · 第6章<br>
<img src="src/zh/svg/chapter_rag_01_rag_flow.svg" width="470" alt="RAG 工作流"><br>
<sub>「有据可依」到底发生在哪一步</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>三层记忆架构</b> · 第4章<br>
<img src="src/zh/svg/chapter_memory_01_memory_types.svg" width="470" alt="记忆架构"><br>
<sub>什么该沉入长期记忆，什么该被捞回来</sub>
</td>
<td width="50%" align="center">
<b>提示工程 vs 上下文工程</b> · 第7章<br>
<img src="src/zh/svg/chapter_context_01_comparison.svg" width="470" alt="上下文工程"><br>
<sub>从「怎么说」到「模型看到了什么」</sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<b>MCP / A2A / ANP 协议栈</b> · 第17章<br>
<img src="src/zh/svg/chapter_protocol_03_three_protocols.svg" width="470" alt="协议对比"><br>
<sub>发现 → 任务协作 → 工具调用</sub>
</td>
<td width="50%" align="center">
<b>GRPO 训练架构</b> · 第10章<br>
<img src="src/zh/svg/chapter_agentic_rl_03_grpo_architecture.svg" width="410" alt="GRPO 架构"><br>
<sub>无需 Critic 模型——用组内归一化算优势</sub>
</td>
</tr>
</table>

### 另有 5 个交互动画

| 动画 | 章节 | 让什么变得可点击 |
| --- | --- | --- |
| 🔄 感知-思考-行动循环 | 第1章 | 单步走完一个完整 Agent 回合 |
| 💡 ReAct 推理过程 | 第5章 | 看思考与行动如何交替 |
| 🔧 Function Calling | 第3章 | 追踪工具调用的往返 |
| 📚 RAG 流程 | 第6章 | 切块 → 向量化 → 检索 → 生成 |
| 🎯 GRPO 采样 | 第10章 | 组内采样与奖励归一化 |

<div align="center"><sub>动画在<a href="https://Haozhe-Xing.github.io/agent_learning/zh/">在线书</a>（以及任何本地构建）中运行。</sub></div>

---

## 📚 完整目录

<details open>
<summary><b>第一部分 — 入门篇</b> · 先建立正确直觉，再写代码</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 1 | [什么是 Agent？](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_intro/) | Agent vs 聊天机器人 vs 传统程序 · 感知-思考-行动循环 · 从符号主义到大模型的历史 |
| 2 | [大语言模型基础](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_llm/) | Tokenizer/BPE · Attention 完整推导 · KV Cache · RoPE · 提示策略 · 模型选型 |

</details>

<details open>
<summary><b>第二部分 — 核心能力篇</b> · 严肃 Agent 必需的八项能力</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 3 | [工具调用 / Function Calling](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_tools/) | 调用机制 · 自定义工具 · 写出模型真会遵守的工具描述 |
| 4 | [记忆系统](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_memory/) | 短期/长期/工作记忆 · 向量检索 · MemGPT & Letta 架构 |
| 5 | [规划与推理](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_planning/) | ReAct · 任务分解 · 反思 · Plan-and-Execute · Test-time Compute Scaling |
| 6 | [RAG 检索增强生成](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_rag/) | 切块 · 向量嵌入 · 重排 · **GraphRAG & Agentic RAG** |
| 7 | [上下文工程](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_context_engineering/) | 注意力预算 · 上下文腐化 · 长程任务策略 · 手写上下文管理器 |
| 8 | [Harness 工程](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_harness/) | 六大工程支柱 · `AGENTS.md` / `CLAUDE.md` · 可靠的结构化输出 |
| 9 | [技能系统](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_skill/) | 技能定义与发现 · 工具 vs 技能 vs 子 Agent · `SKILL.md` 生态 |
| 10 | [Agentic-RL](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_agentic_rl/) | SFT+LoRA · DP/TP/PP/ZeRO · **PPO vs DPO vs GRPO** · 专为 Agent 的微调 |

</details>

<details>
<summary><b>第三部分 — 框架实战与自我进化</b> · 有意识地选工具</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 11 | [自我进化 Agent](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_self_evolving/) | 自动提示优化（APE/OPRO/DSPy/GEPA）· **Agentic 数据飞轮** |
| 12 | [LangChain 深入](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_langchain/) | 超越快速上手——抽象在哪帮你、在哪碍事 |
| 13 | [LangGraph](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_langgraph/) | State/Node/Edge · 为什么状态机比 while 循环更适合真实 Agent |
| 14 | [主流框架全景](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_frameworks/) | CrewAI · AutoGen · Semantic Kernel · 到底该怎么选 |
| 15 | [Claude Code 深度解析](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_claude_code/) | 从使用到内部机制 · 终端 Agent 最佳实践 |

</details>

<details>
<summary><b>第四部分 — 多 Agent 系统篇</b> · 当一个 Agent 不够用</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 16 | [多 Agent 协作](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_multi_agent/) | Supervisor vs 去中心化 · 消息队列 / 黑板 / 直接调用 |
| 17 | [通信协议](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_protocol/) | **MCP · A2A · ANP** —— 正在形成的三层协议栈 |

</details>

<details>
<summary><b>第五部分 — 生产化篇</b> · 教程们跳过的那部分</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 18 | [评估与优化](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_evaluation/) | GAIA · SWE-bench · LLM-as-Judge 校准 · 可观测性 · 成本 |
| 19 | [安全与可靠性](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_security/) | **提示注入**（直接与间接）· 权限沙箱 · fail-closed 设计 |
| 20 | [部署与生产化](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_deployment/) | API 服务 · 流式输出 · 容器化 · 扩缩容 |

</details>

<details>
<summary><b>第六部分 — 综合项目篇</b> + 附录</summary>

| 章 | 标题 | 重点 |
| --- | --- | --- |
| 21 | [AI 编程助手](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_coding_agent/) | 读仓库、改文件、跑测试、自修复循环 |
| 22 | [智能数据分析 Agent](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_data_agent/) | 沙箱代码执行 · Pipeline vs Agentic 循环 |
| 23 | [多模态 Agent](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_multimodal/) | 视觉 + 文本的工具使用 |

**附录：**[提示词模板](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/prompt_templates.html) · [FAQ](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/faq.html) · [学习资源](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/resources.html) · [术语表](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/glossary.html) · [KL 散度详解](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/kl_divergence.html) · [环境搭建](https://Haozhe-Xing.github.io/agent_learning/zh/chapter_setup/)

</details>

---

## ⚡ 快速开始

**只想读？** → [**打开在线书**](https://Haozhe-Xing.github.io/agent_learning/zh/)，什么都不用装。

<details>
<summary><b>本地构建</b>（mdBook）</summary>

```bash
# 1. 安装 mdBook + KaTeX 插件
cargo install mdbook mdbook-katex
# macOS 也可以：brew install mdbook

# 2. 克隆并启动（同时构建中英双版，端口 3000）
git clone https://github.com/Haozhe-Xing/agent_learning.git
cd agent_learning
./serve.sh
```

| 地址 | 内容 |
| --- | --- |
| `http://localhost:3000` | 语言选择页 |
| `http://localhost:3000/zh/` | 中文版 |
| `http://localhost:3000/en/` | English |

</details>

<details>
<summary><b>跑代码</b>（Python 3.11+）</summary>

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install langchain langchain-openai langgraph openai anthropic
export OPENAI_API_KEY="sk-..."
```

仓库里还有 **`reference-agent/`**——一个依赖极少的最小 Agent 实现（主循环、工具注册表、记忆、权限闸门），作为实战章节共用的教学底座。当框架抽象开始像魔法时，来读它。

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

## ❓ 常见问题

<details>
<summary><b>需要机器学习背景吗？</b></summary>

不需要。第 1–9 章只要会 Python。数学出现在第 10 章（Agentic-RL），配 KaTeX 渲染的推导，而且即便在那里也是先讲直觉再上公式——另有[附录 E](https://Haozhe-Xing.github.io/agent_learning/zh/appendix/kl_divergence.html) 从零讲 KL 散度。如果你只想做应用，第 10–11 章可以整章跳过。

</details>

<details>
<summary><b>LangChain 一改 API，这书是不是就废了？</b></summary>

基本不会——这是刻意设计的。书讲的是*机制*（为什么记忆要分层、为什么上下文会腐化、为什么工具描述本身就是提示词），框架只被当作这些机制的实现。需要维护的是框架章节（12–15），概念主干不需要。

</details>

<details>
<summary><b>和论文清单有什么区别？</b></summary>

这里的论文永远挂在一个工程问题上。每篇都写清当时的问题、机制、**对你代码的启示**、以及局限——然后用对比表把一组论文综合起来。目标是让你能*用*这个想法，不只是引用它。

</details>

<details>
<summary><b>英文版是机器翻译的吗？</b></summary>

本书是双语撰写、两版同步维护（`src/zh/` 与 `src/en/`），包含各自独立的图解。两者都不是对方的原始机翻。

</details>

<details>
<summary><b>「持续追踪 arXiv」具体是什么意思？</b></summary>

扫描 arXiv 上新的 Agent 相关工作（架构、记忆、工具、多 Agent、Agent 强化学习、安全），把相关论文消化后写入对应章节的「最新进展」小节。实际节奏是每隔几天一批，而非严格每 24 小时——具体可看[提交历史](https://github.com/Haozhe-Xing/agent_learning/commits/main)。重点在于前沿部分会持续生长，不必等改版。

</details>

---

## 🤝 参与贡献

各种形式的贡献都欢迎——改错别字也算。

| | |
| --- | --- |
| 🐛 发现错误 | [提 Issue](https://github.com/Haozhe-Xing/agent_learning/issues) |
| 💡 某章看不懂 | [告诉我们在哪](https://github.com/Haozhe-Xing/agent_learning/issues/new) —— 讲不清楚也是 bug |
| 📝 想改进内容 | Fork → 编辑 → PR |
| 🌍 修订译文 | `src/zh/` 和 `src/en/` 都欢迎改进 |

```bash
git clone https://github.com/YOUR_USERNAME/agent_learning.git
git checkout -b feature/improve-chapter-3
./serve.sh                # 本地预览
git commit -m "feat: 优化第3章工具描述示例"
```

<details>
<summary><b>仓库约定</b></summary>

- 章节放在 `src/zh/chapter_xxx/` 和 `src/en/chapter_xxx/`
- 章节概览写进 `README.md`；小节按 `01_xxx.md`、`02_xxx.md` 编号
- 图解放 `src/{zh,en}/svg/`，命名 `chapter_<名称>_<描述>.svg`
- 动画放 `src/{zh,en}/animations/`
- 新增页面要在**两种语言**的 `SUMMARY.md` 里都登记

</details>

<details>
<summary><b>论文解读模板</b></summary>

前沿小节统一结构，让读者能快速判断相关性：

```markdown
### 论文标题：一句话说清它解决的问题

- **论文链接**：
- **代码 / 项目链接**：
- **年份 / 机构**：
- **当时要解决的问题**：
- **核心贡献**：
- **方法拆解**：
- **对 Agent 工程的启示**：
- **局限**：
```

质量要求：链到一手来源 · 讲清*历史*贡献而非复述摘要 · 关联到记忆/工具/规划/评估/安全/训练 · 说明它**没**解决什么 · 几篇之后要加综合对比表，而不是留一串清单。

</details>

---

## 🗺️ 路线图

**已完成** — ✅ 中英双语 mdBook · ✅ 336 张原创图解 · ✅ 交互动画 · ✅ 核心章节论文精读 · ✅ Agentic-RL（PPO/DPO/GRPO）· ✅ `reference-agent/` 教学底座 · ✅ 持续追踪 arXiv 论文

**下一步** — ⬜ 更多可运行的端到端项目模板 · ⬜ Agent 关键词速查表 · ⬜ 图解画廊索引 · ⬜ 面试题与自测练习 · ⬜ 接好评估与可观测性的生产模板

有需求？[提个 Issue](https://github.com/Haozhe-Xing/agent_learning/issues/new) —— 路线图由读者驱动。

---

<div align="center">

### ⭐ 如果它帮你省了时间，点个 Star

不是为了虚荣指标——Star 决定了下一个卡在上下文腐化、
提示注入、或者「到底选哪个框架」的工程师，能找到这个仓库，而不是一份 500 链接的 awesome 列表。

<a href="https://github.com/Haozhe-Xing/agent_learning"><img src="https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=for-the-badge&logo=github&color=FFD700&label=Star%20this%20repo" alt="Star"></a>

[![Star History Chart](https://api.star-history.com/svg?repos=Haozhe-Xing/agent_learning&type=Date)](https://www.star-history.com/#Haozhe-Xing/agent_learning&Date)

<br>

**[📖 开始阅读](https://Haozhe-Xing.github.io/agent_learning/zh/)** · **[🇺🇸 English](https://Haozhe-Xing.github.io/agent_learning/en/)** · **[🐛 Issues](https://github.com/Haozhe-Xing/agent_learning/issues)** · **[📄 MIT License](LICENSE)**

<sub>写它，是为了让理解 Agent 不必先读 200 篇论文和 12 份框架变更日志。</sub>

</div>
