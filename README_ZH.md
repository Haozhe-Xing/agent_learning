<div align="center">

<img src="readme_img.png" width="900" alt="Agent 学习路线图">

<br>

# 🤖 Agent Learning：从零开始学 Agent

**一份从 LLM 基础到生产级 Agent 系统的完整开源学习路线图。**

**Agent Learning**（`agent_learning`）是一个系统、实战导向的 AI Agent 学习路线图与动手教程，覆盖 LLM 基础、RAG、记忆系统、工具调用、Function Calling、Agentic Workflow、LangChain、LangGraph、MCP、多智能体系统、评估、部署和 Agentic RL。

> 如果你想系统学习如何构建 AI Agent，而不只是会用 ChatGPT，本项目会从 LLM 基础讲到 RAG、Memory、Tool Use、Planning、多 Agent、LangGraph、MCP、部署评估与生产级 Agent 系统。

**每日自动追踪 arXiv 前沿论文，内容持续更新，始终紧跟最新进展**

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/Haozhe-Xing/agent_learning?style=social)](https://github.com/Haozhe-Xing/agent_learning)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Haozhe-Xing/agent_learning/pulls)
[![mdBook](https://img.shields.io/badge/built%20with-mdBook-blue)](https://rust-lang.github.io/mdBook/)
[![Daily arXiv](https://img.shields.io/badge/每日-arXiv%20更新-red?logo=arxiv)](https://arxiv.org)

<br>

[<img src="https://img.shields.io/badge/📖_在线阅读（中文）-4CAF50?style=for-the-badge" alt="在线阅读中文版">](https://Haozhe-Xing.github.io/agent_learning/zh/)&nbsp;&nbsp;&nbsp;[<img src="https://img.shields.io/badge/📖_Read_Online_(English)-2196F3?style=for-the-badge" alt="Read Online English">](https://Haozhe-Xing.github.io/agent_learning/en/)

<br>

<table>
<tr>
<td width="50%" align="center">
<img src="scripts/agent2.png" width="100%" alt="Agent Learning 在线书截图 - 前沿研究章节">
<br>
<sub>每日更新的 Agentic-RL 前沿研究章节</sub>
</td>
<td width="50%" align="center">
<img src="scripts/grpo_chapter.png" width="100%" alt="Agent Learning 在线书截图 - GRPO 章节">
<br>
<sub>循序渐进的 GRPO / GSPO 学习内容</sub>
</td>
</tr>
</table>

<br>

[🐛 提交问题](https://github.com/Haozhe-Xing/agent_learning/issues) · [💬 参与讨论](https://github.com/Haozhe-Xing/agent_learning/discussions) · [🇺🇸 English README](README.md)

</div>

---

## 🚀 自动追踪前沿：每日 arXiv 论文更新

<div align="center">

🤖 **本仓库每天自动搜索 arXiv 上最新的 AI Agent 相关论文，并将内容自动更新到对应章节 —— 确保你始终紧跟研究前沿！**

</div>

- 📡 **每日自动检索**：定时流水线每天扫描 arXiv 上关于 Agent 架构、工具使用、记忆系统、多智能体协作、强化学习等方向的最新论文。
- 📝 **内容自动更新**：相关研究成果自动整合到对应章节，保持前沿内容的时效性与新鲜度。
- 🔔 **不错过任何突破**：无需手动追踪大量研究源，本仓库自动完成，让你专注于学习和构建。

> 💡 这意味着你在这里阅读的内容**不是静态的** —— 它会随着 AI Agent 领域的最新进展持续演进。

---

## 👥 适合谁阅读？

- **开发者**：想真正构建 AI Agent 应用，而不只是写 Prompt 调聊天机器人
- **学生和初学者**：需要一条从 LLM 基础到 Agent 系统的系统学习路径
- **LLM 应用工程师**：正在实践 RAG、工具调用、记忆系统、LangGraph、MCP 和评估
- **研究者和技术创作者**：希望把前沿 Agent 论文和工程实践连接起来
- **产品和创业团队**：想理解生产级 Agent 工作流如何设计与落地

---

## 🧭 学习路径

如果你不知道从哪里开始，可以直接按下面的目标选择阅读顺序。

| 你的目标 | 推荐阅读路线 |
| -------- | ------------ |
| **快速理解 Agent 是什么** | 第 1 章 Agent 基础 → 第 2 章大语言模型基础 → 第 3 章工具调用 → 第 5.2 节 ReAct → 附录 F.4 Hello Agent |
| **了解大模型，并学习使用 Codex、Claude Code 等 AI 编程工具** | 第 2 章大语言模型基础 → 第 2.2 节 Prompt Engineering → 第 2.4 节模型 API 调用入门 → 第 14 章 Claude Code 深度解析 → 第 20 章 AI 编程助手 |
| **构建一个能用的 Agent 应用** | 第 3 章工具调用 → 第 4 章记忆系统 → 第 5 章规划与推理 → 第 6 章 RAG → 第 7 章上下文工程 → 第 8 章 Harness Engineering |
| **把 Agent 上线到生产环境** | 第 8 章 Harness Engineering → 第 17 章评估与优化 → 第 18 章安全与可靠性 → 第 19 章部署与生产化 |
| **掌握 LangChain / LangGraph 等框架** | 第 3 章工具调用 → 第 5 章规划与推理 → 第 6 章 RAG → 第 11 章 LangChain → 第 12 章 LangGraph → 第 13 章框架概览 |
| **理解 Agent 前沿论文和研究脉络** | 第 5.2 节 ReAct → 第 5.4 节反思机制 → 第 5.7 节规划论文解读 → 第 4.6 节记忆论文解读 → 第 6.6 节 RAG 论文解读 → 第 15.6 节多 Agent 论文解读 |
| **学习 Agentic-RL 和自我进化** | 第 2.8 节训练数据准备 → 第 10.1 节 Agentic-RL → 第 10.4 节 DPO → 第 10.5 节 GRPO / GSPO → 第 10.8 节 Agent 微调 → 第 10.9 节数据飞轮 → 第 10.10 节 Self-Evolution Agent |
| **边做项目边学习** | 附录 F 开发环境搭建 → 第 3 章工具调用 → 第 6 章 RAG → 第 12 章 LangGraph → 第 20 章 AI 编程助手 → 第 21 章数据分析 Agent → 第 22 章多模态 Agent |

---

## ✨ 本书特色

- 🎯 **循序渐进**：从 LLM 基础到多 Agent 系统，每章都有清晰的知识脉络
- 💻 **代码优先**：每个核心概念都配有可运行的 Python 代码示例
- 🎨 **图文并茂**：120+ 手绘 SVG 架构图 / 流程图 / 时序图，直观理解复杂概念
- 🎬 **交互动画**：内置 5 个交互式 HTML 动画（感知-思考-行动循环、ReAct 推理、Function Calling、RAG 流程、GRPO 采样）
- 🔬 **论文解读**：关键章节附有前沿论文精读（ReAct、Reflexion、MemGPT、GRPO 等），帮你跟上学术最新进展
- 🏗️ **完整项目**：3 个综合实战项目（AI 编程助手、智能数据分析 Agent、多模态 Agent）
- 🛡️ **生产就绪**：涵盖安全、评估、部署等生产环境必备知识
- 🧪 **前沿技术**：涵盖上下文工程、Agentic-RL（GRPO/DPO/PPO）、MCP/A2A/ANP 等 2025—2026 最新进展
- 📐 **公式支持**：使用 KaTeX 渲染数学公式，强化学习章节可清晰阅读策略梯度、KL 散度等公式推导
- 🔄 **持续更新**：跟踪 LangChain、LangGraph、MCP 等框架的最新变化

---

## 📸 内容精选预览

> 以下是本书 **120+ 手绘 SVG 插图**中的精选展示，所有图示均为本书原创。

### 🧠 Agent 核心架构

<table>
<tr>
<td width="50%" align="center">

**感知-思考-行动循环（第 1 章）**

<img src="src/zh/svg/chapter_intro_03_loop.svg" width="420" alt="感知-思考-行动循环">

<sub>Agent 的核心运行机制：感知环境 → LLM 推理决策 → 执行行动 → 循环直到目标达成</sub>

</td>
<td width="50%" align="center">

**ReAct 推理框架（第 5 章）**

<img src="src/zh/svg/chapter_planning_02_react_loop.svg" width="420" alt="ReAct 推理框架">

<sub>Thought → Action → Observation 交替循环，让 Agent 边思考边行动</sub>

</td>
</tr>
</table>

### 🛠️ 工具调用与 RAG

<table>
<tr>
<td width="50%" align="center">

**Function Calling 完整流程（第 3 章）**

<img src="src/zh/svg/chapter_tools_02_function_calling.svg" width="480" alt="Function Calling 流程">

<sub>从用户输入到工具调用再到最终回复的 6 步完整流程，附消息结构示意</sub>

</td>
<td width="50%" align="center">

**RAG 检索增强生成（第 6 章）**

<img src="src/zh/svg/chapter_rag_01_rag_flow.svg" width="480" alt="RAG 工作原理">

<sub>离线建库 + 在线检索双阶段架构，让 LLM 回答有据可查</sub>

</td>
</tr>
</table>

### 💾 记忆系统与上下文工程

<table>
<tr>
<td width="50%" align="center">

**记忆系统三层架构（第 4 章）**

<img src="src/zh/svg/chapter_memory_01_memory_types.svg" width="480" alt="记忆系统三层架构">

<sub>工作记忆 → 短期记忆 → 长期记忆，重要信息向下沉淀、语义检索向上提取</sub>

</td>
<td width="50%" align="center">

**提示工程 vs 上下文工程（第 7 章）**

<img src="src/zh/svg/chapter_context_01_comparison.svg" width="480" alt="提示工程 vs 上下文工程">

<sub>从"如何说"到"让 LLM 看到什么"——Agent 时代的范式升级</sub>

</td>
</tr>
</table>

### 🤝 多 Agent 与通信协议

<table>
<tr>
<td width="50%" align="center">

**多 Agent 三种通信模式（第 15 章）**

<img src="src/zh/svg/chapter_multi_agent_02_communication.svg" width="480" alt="多 Agent 通信模式">

<sub>消息队列（异步解耦）/ 共享黑板（数据共享）/ 直接调用（实时协作）</sub>

</td>
<td width="50%" align="center">

**MCP / A2A / ANP 三协议对比（第 16 章）**

<img src="src/zh/svg/chapter_protocol_03_three_protocols.svg" width="480" alt="三协议对比">

<sub>三层协议栈各司其职：ANP 组网发现 → A2A 任务协作 → MCP 工具调用</sub>

</td>
</tr>
</table>

### 🧪 强化学习与框架

<table>
<tr>
<td width="50%" align="center">

**GRPO 训练架构（第 10 章）**

<img src="src/zh/svg/chapter_agentic_rl_03_grpo_architecture.svg" width="420" alt="GRPO 训练架构">

<sub>无需 Critic 模型，通过组内标准化计算优势值，显存仅需 1.5× 模型大小</sub>

</td>
<td width="50%" align="center">

**LangGraph 三大核心概念（第 12 章）**

<img src="src/zh/svg/chapter_langgraph_02_state_node_edge.svg" width="480" alt="LangGraph 核心概念">

<sub>State（共享状态）· Node（处理单元）· Edge（执行流控制）</sub>

</td>
</tr>
</table>

<div align="center">

📖 **以上仅为精选预览** — 完整的 120+ 张架构图 + 5 个交互动画，请 [**在线阅读**](https://Haozhe-Xing.github.io/agent_learning) 体验

</div>

---

## 🎬 交互式动画

本书内置了 **5 个可交互的 HTML 动画**，帮助你直观理解核心概念的动态过程：

| 动画 | 对应章节 | 说明 |
|------|----------|------|
| 🔄 **感知-思考-行动循环** | 第 1 章 | 动态演示 Agent 的核心运行循环 |
| 💡 **ReAct 推理过程** | 第 5 章 | 展示 Thought → Action → Observation 的交替过程 |
| 🔧 **Function Calling** | 第 3 章 | 工具调用的完整流程动画 |
| 📚 **RAG 检索流程** | 第 6 章 | 从文档切分到向量检索再到生成回答 |
| 🎯 **GRPO 采样过程** | 第 10 章 | 组内多输出采样与奖励标准化的可视化 |

> 💡 交互动画仅在 [在线电子书](https://Haozhe-Xing.github.io/agent_learning) 中可体验，本地构建也可预览。

---

## 🔥 核心知识点速览

<table>
<tr>
<td width="50%">

**🧠 Agent 核心架构**
- 感知 → 思考 → 行动循环
- ReAct 推理框架
- 任务分解与规划
- 反思与自我纠错

**🛠️ 工具与技能**
- Function Calling 机制
- 自定义工具设计
- 技能系统构建
- 工具描述最佳实践

**🧪 强化学习训练**
- SFT + LoRA 基础训练
- PPO / DPO / GRPO 算法详解
- 完整训练 Pipeline 实战
- 2025—2026 最新研究进展

</td>
<td width="50%">

**💾 记忆、知识与上下文**
- 短期 / 长期 / 工作记忆
- 向量数据库（Chroma / FAISS）
- RAG 检索增强生成
- 上下文工程与注意力预算

**🤝 多 Agent 协作 & 通信**
- MCP / A2A / ANP 三协议栈
- Supervisor vs 去中心化模式
- CrewAI / AutoGen 框架
- LangGraph 有状态 Agent

**🛡️ 生产化全链路**
- 评估基准（GAIA / SWE-bench）
- 安全防御与沙箱隔离
- 容器化部署与流式响应
- 可观测性与成本优化

</td>
</tr>
</table>

---

## 🚀 快速开始

### 本地构建

```bash
# 安装 mdBook（二选一）
cargo install mdbook
# 或 macOS：brew install mdbook

# 安装 mdbook-katex 插件（用于数学公式渲染）
cargo install mdbook-katex

# 克隆仓库
git clone https://github.com/Haozhe-Xing/agent_learning.git
cd agent_learning

# 构建中英文版本并启动统一服务（默认端口 3000）
./serve.sh
```

启动后访问：
- 🌐 **语言选择首页**：`http://localhost:3000`
- 🇨🇳 **中文版**：`http://localhost:3000/zh/`
- 🇺🇸 **English**：`http://localhost:3000/en/`

### 环境准备（跟随代码实践）

```bash
# Python 3.11+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装核心依赖
pip install langchain langchain-openai langgraph openai anthropic

# 配置 API Key
export OPENAI_API_KEY="your-key-here"
```

---

## 📊 涵盖技术栈

![Python](https://img.shields.io/badge/Python_3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic_(Claude)-191919?style=flat)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat)
![mdBook](https://img.shields.io/badge/mdBook-000000?style=flat&logo=rust&logoColor=white)
![KaTeX](https://img.shields.io/badge/KaTeX-44CC11?style=flat)

---

## 🤝 参与贡献

欢迎任何形式的贡献！

- 🐛 **发现错误**：[提交 Issue](https://github.com/Haozhe-Xing/agent_learning/issues)
- 💡 **内容建议**：[发起 Discussion](https://github.com/Haozhe-Xing/agent_learning/discussions)
- 📝 **改进内容**：Fork → 修改 → 提交 PR
- ⭐ **支持项目**：给本仓库点个 Star！

### 贡献指南

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/agent_learning.git

# 创建特性分支
git checkout -b feature/improve-chapter-3

# 本地预览
./serve.sh

# 提交并推送
git commit -m "feat: 改进第3章工具调用示例代码"
git push origin feature/improve-chapter-3
```

### 内容组织约定

- 每章内容放在独立目录 `src/zh/chapter_xxx/`（中文）或 `src/en/chapter_xxx/`（英文）下
- 章节概述放在 `README.md`，各小节按 `01_xxx.md`、`02_xxx.md` 编号
- 中文版 SVG 插图放在 `src/zh/svg/`，英文版放在 `src/en/svg/`，命名格式 `chapter_xxx_描述.svg`
- 中文版交互动画放在 `src/zh/animations/`，英文版放在 `src/en/animations/`

### 论文解读模板

所有论文解读和前沿进展章节应采用统一结构，帮助读者快速理解一篇论文为什么重要、当时解决了什么问题，以及它如何影响真实 Agent 工程。

每篇代表性论文建议使用以下模板：

```markdown
### 论文名：一句话说明它解决什么问题

- **论文链接**：
- **代码 / 项目链接**：
- **发表时间 / 机构**：
- **当时解决的问题**：
- **核心贡献**：
- **方法拆解**：
- **对 Agent 工程的启发**：
- **局限性**：
```

质量要求：

- **链接到原始来源**：优先提供 arXiv、会议页面、官方博客、GitHub 或项目主页。
- **说明历史贡献**：不要只说论文做了什么，还要说明它在发表时解决了什么关键问题。
- **连接工程实践**：说明该工作对 Agent 记忆、工具、规划、评估、安全、训练或部署有什么启发。
- **写清局限性**：说明论文没有解决什么、依赖哪些强假设，或者是否主要是 benchmark 驱动。
- **避免只堆论文列表**：多篇论文之后，应补充对比表或脉络小结，说明这些工作之间的关系。

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🗺️ 项目路线图

- [x] 基于 mdBook 的中英文在线书
- [x] 120+ 原创 SVG 架构图和流程图
- [x] Agent 核心概念交互式动画
- [x] 关键 Agent 论文精读章节
- [x] 覆盖 PPO / DPO / GRPO 的 Agentic RL 内容
- [ ] 可运行 Agent 示例项目和模板
- [ ] Agent 术语表与关键词速查表
- [ ] Agent 架构图合集
- [ ] 面试题和自测题
- [ ] 带评估与可观测性的生产级 Agent 模板

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐，这是对作者最大的鼓励！

[![Star History Chart](https://api.star-history.com/svg?repos=Haozhe-Xing/agent_learning&type=Date)](https://www.star-history.com/#Haozhe-Xing/agent_learning&Date)

---

<div align="center">

**用 ❤️ 构建，为了让每个开发者都能掌握 AI Agent 开发**

[⬆ 回到顶部](#-从零开始学-agent)

</div>