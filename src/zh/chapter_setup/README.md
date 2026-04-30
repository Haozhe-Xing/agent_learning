# 附录 F：开发环境搭建

> 磨刀不误砍柴工。一个配置良好的开发环境，能让你的 Agent 开发事半功倍。

---

## 本章概览

本章从零开始，帮你搭建一个专业的 Agent 开发环境。从 Python 虚拟环境管理，到 API Key 安全存储，再到第一个可运行的 Agent——跟着本章操作，你将拥有一个随时可以开始开发的工作台。

## 本章目标

学完本章，你将能够：

- ✅ 使用 `uv` 或 `conda` 管理 Python 环境和依赖
- ✅ 安装并验证 LangChain、OpenAI SDK 等核心库
- ✅ 安全管理 API Key，避免密钥泄露
- ✅ 运行你的第一个真正的 Agent！

## 本章结构

| 小节 | 内容 | 预计时间 |
|------|------|---------|
| 2.1 Python 环境与依赖管理 | venv、conda、uv 对比与使用 | 20分钟 |
| 2.2 关键库安装 | LangChain、OpenAI SDK 等 | 15分钟 |
| 2.3 API Key 管理 | .env 文件、环境变量最佳实践 | 10分钟 |
| 2.4 Hello Agent！ | 第一个完整的 Agent | 30分钟 |

## 前置要求

- 已安装 Python 3.10 或更高版本
- 有基本的命令行操作经验
- 已注册 OpenAI 账户（或其他 LLM 服务）

## 工具选择建议

| 工具 | 用途 | 推荐程度 |
|------|------|---------|
| `uv` | Python 包管理（最新最快） | ⭐⭐⭐⭐⭐ 强烈推荐 |
| `conda` | 环境管理（数据科学常用） | ⭐⭐⭐⭐ 推荐 |
| `venv` | 内置虚拟环境 | ⭐⭐⭐ 够用 |
| VS Code | 代码编辑器 | ⭐⭐⭐⭐⭐ 推荐 |
| Jupyter | 交互式开发和实验 | ⭐⭐⭐⭐ 推荐 |

## 🔗 学习路径

> **前置知识**：[第1章 什么是 Agent？](../chapter_intro/README.md)
>
> **后续推荐**：
> - 👉 [第2章 大语言模型基础](../chapter_llm/README.md) — 理解 Agent 的核心"大脑"
> - 👉 [第3章 工具调用（Tool Use / Function Calling）](../chapter_tools/README.md) — 让 Agent 能"动手做事"

---

*下一节：[F.1 Python 环境与依赖管理](./01_python_setup.md)*
