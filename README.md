<div align="center">

# 🤖 从零开始学 Agent 开发

**一本系统、全面、实战导向的 AI Agent 开发教程**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/yourusername/agent_learning?style=social)](https://github.com/yourusername/agent_learning)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/agent_learning/pulls)
[![mdBook](https://img.shields.io/badge/built%20with-mdBook-blue)](https://rust-lang.github.io/mdBook/)

[📖 在线阅读](https://yourusername.github.io/agent_learning) · [🐛 提交问题](https://github.com/yourusername/agent_learning/issues) · [💬 参与讨论](https://github.com/yourusername/agent_learning/discussions)

</div>

---

## 📌 为什么写这本书？

AI Agent 正在重塑软件开发的边界。从 GitHub Copilot 到 Devin，从 AutoGPT 到 Claude，**会构建 Agent 的工程师正在成为最稀缺的技术人才**。

然而，现有的学习资源要么过于零散，要么停留在理论层面，缺乏一条从入门到生产的完整路径。

这本书的目标只有一个：**让你真正能构建出可用的 AI Agent 系统**。

> 📚 本书已构建为在线电子书，支持全文搜索、暗色模式，可直接在浏览器中阅读。

---

## ✨ 本书特色

- 🎯 **循序渐进**：从 LLM 基础到多 Agent 系统，每章都有清晰的知识脉络
- 💻 **代码优先**：每个核心概念都配有可运行的 Python 代码示例
- 🔬 **论文解读**：关键章节附有前沿论文精读，帮你跟上学术最新进展
- 🏗️ **完整项目**：3 个综合实战项目（编程助手、数据分析 Agent、多模态 Agent）
- 🛡️ **生产就绪**：涵盖安全、评估、部署等生产环境必备知识
- 🔄 **持续更新**：跟踪 LangChain、LangGraph、MCP 等框架的最新变化

---

## 📚 内容目录

<details>
<summary><b>第一部分：基础概念篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第1章 | 什么是 Agent？从聊天机器人到智能体的演进、核心概念、架构与应用场景 |
| 第2章 | 大语言模型基础：LLM 工作原理、Prompt Engineering、CoT 提示策略、API 调用 |
| 第3章 | 开发环境搭建：Python 环境、关键库安装、API Key 管理、第一个 Hello Agent |

</details>

<details>
<summary><b>第二部分：核心技术篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第4章 | 工具调用（Tool Use / Function Calling）：机制详解、自定义工具、论文解读 |
| 第5章 | Skill System：技能定义、学习、发现与注册，构建可复用技能系统 |
| 第6章 | 记忆系统（Memory）：短期/长期/工作记忆，向量数据库，论文解读 |
| 第7章 | 规划与推理（Planning & Reasoning）：ReAct 框架、任务分解、反思机制 |
| 第8章 | 检索增强生成（RAG）：文档加载、向量嵌入、检索策略、论文解读 |

</details>

<details>
<summary><b>第三部分：框架实战篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第9章 | LangChain 深入实战：Chain、Agent、LCEL 表达式语言 |
| 第10章 | LangGraph：构建有状态的 Agent，条件路由、Human-in-the-Loop |
| 第11章 | 其他主流框架：AutoGPT、CrewAI、AutoGen、Dify/Coze，如何选型 |

</details>

<details>
<summary><b>第四部分：多 Agent 系统篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第12章 | 多 Agent 协作：通信模式、角色分工、Supervisor vs 去中心化 |
| 第13章 | Agent 通信协议：MCP 协议详解、A2A 协议、消息传递与状态共享 |

</details>

<details>
<summary><b>第五部分：高级主题篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第14章 | Agent 评估与优化：评估方法、基准测试、Prompt 调优、成本控制、可观测性 |
| 第15章 | 安全与可靠性：Prompt 注入防御、幻觉问题、权限控制、数据保护、行为对齐 |
| 第16章 | 部署与生产化：部署架构、FastAPI 封装、容器化、流式响应、生产级实战 |

</details>

<details>
<summary><b>第六部分：综合项目篇</b></summary>

| 章节 | 内容 |
|------|------|
| 第17章 | 🔨 项目实战：AI 编程助手（代码理解、生成、测试、Bug 修复） |
| 第18章 | 📊 项目实战：智能数据分析 Agent（数据连接、分析、可视化、报告生成） |
| 第19章 | 🎨 项目实战：多模态 Agent（图像理解、语音交互、多模态个人助理） |

</details>

---

## 🗺️ 学习路线图

```
入门阶段                核心技术                框架实战               进阶生产
─────────             ─────────              ─────────             ─────────
第1章 Agent概念  →    第4章 工具调用    →    第9章 LangChain  →   第14章 评估优化
第2章 LLM基础    →    第5章 技能系统    →    第10章 LangGraph →   第15章 安全可靠
第3章 环境搭建   →    第6章 记忆系统    →    第11章 框架选型  →   第16章 部署生产
                      第7章 规划推理    →    第12章 多Agent   →   第17-19章 综合项目
                      第8章 RAG        →    第13章 通信协议
```

---

## 🚀 快速开始

### 在线阅读（推荐）

直接访问 👉 **[https://yourusername.github.io/agent_learning](https://yourusername.github.io/agent_learning)**

### 本地构建

```bash
# 安装 mdBook
cargo install mdbook

# 克隆仓库
git clone https://github.com/yourusername/agent_learning.git
cd agent_learning

# 本地预览
mdbook serve src --open
```

### 环境准备（跟随代码实践）

```bash
# Python 3.10+
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装核心依赖
pip install langchain langchain-openai langgraph openai anthropic

# 配置 API Key
export OPENAI_API_KEY="your-key-here"
```

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

</td>
<td width="50%">

**💾 记忆与知识**
- 短期/长期/工作记忆
- 向量数据库（Chroma/Pinecone）
- RAG 检索增强生成
- 知识图谱集成

**🤝 多 Agent 协作**
- MCP 协议
- A2A 通信
- Supervisor 模式
- CrewAI / AutoGen

</td>
</tr>
</table>

---

## 📊 涵盖技术栈

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Chroma](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat)
![LangGraph](https://img.shields.io/badge/LangGraph-1C3C3C?style=flat)

---

## 🤝 参与贡献

欢迎任何形式的贡献！

- 🐛 **发现错误**：[提交 Issue](https://github.com/yourusername/agent_learning/issues)
- 💡 **内容建议**：[发起 Discussion](https://github.com/yourusername/agent_learning/discussions)
- 📝 **改进内容**：Fork → 修改 → 提交 PR
- ⭐ **支持项目**：给本仓库点个 Star！

### 贡献指南

```bash
# Fork 并克隆
git clone https://github.com/YOUR_USERNAME/agent_learning.git

# 创建特性分支
git checkout -b feature/improve-chapter-4

# 提交修改
git commit -m "feat: 改进第4章工具调用示例代码"

# 推送并创建 PR
git push origin feature/improve-chapter-4
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## ⭐ Star History

如果这个项目对你有帮助，请给个 Star ⭐，这是对作者最大的鼓励！

---

<div align="center">

**用 ❤️ 构建，为了让每个开发者都能掌握 AI Agent 开发**

[⬆ 回到顶部](#-从零开始学-agent-开发)

</div>