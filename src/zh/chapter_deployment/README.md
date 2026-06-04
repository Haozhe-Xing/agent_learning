# 第19章 部署与生产化

> 代码在本地跑通只是第一步。让 Agent 在生产环境稳定服务用户，才是真正的考验。

---

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 掌握 Agent 应用的部署架构设计原则
- ✅ 用 FastAPI 将 Agent 封装为 REST API 服务
- ✅ 实现 Docker 容器化和基本云部署流程
- ✅ 处理流式响应和高并发场景
- ✅ 完成一个生产级 Agent 服务的完整部署

## ⏱️ 预计学习时间

约 **120-150 分钟**（含实战）

## 💡 前置知识

- 完成第17章（评估与优化）和第18章（安全与可靠性）
- 了解 REST API 和 HTTP 协议
- 有 Linux 基础命令行操作经验，了解 Docker 基本概念（不要求精通）

## 📚 本章结构

本章覆盖 Agent 从开发环境到生产环境的完整路径：部署架构设计、API 服务化、Docker 容器化、流式响应与并发处理，最终构建一个完整的生产级 Agent 服务。

| 小节 | 内容 |
|------|------|
| 19.1 Agent 应用的部署架构 | 分层架构、状态管理 |
| 19.2 API 服务化 | FastAPI 封装、SSE 流式 |
| 19.3 容器化与云部署 | Dockerfile、Docker Compose |
| 19.4 流式响应与并发处理 | 异步、信号量、队列 |
| 19.5 实战：生产级 Agent 服务 | 完整部署流程 |

## 🔗 学习路径

> **前置知识**：[第17章 Agent 的评估与优化](../chapter_evaluation/README.md)、[第18章 安全与可靠性](../chapter_security/README.md)
>
> **后续推荐**：
> - 👉 [第20章 项目实战：AI 编程助手](../chapter_coding_agent/README.md) — 综合项目实战
> - 👉 [第21章 项目实战：智能数据分析 Agent](../chapter_data_agent/README.md) — 综合项目实战

---

*下一节：[19.1 Agent 应用的部署架构](./01_deployment_architecture.md)*
