# 第20章 部署与生产化

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

- 完成第18章（评估与优化）和第19章（安全与可靠性）
- 了解 REST API 和 HTTP 协议
- 有 Linux 基础命令行操作经验

---

## 🔗 学习路径

> **后续推荐**：
> - 👉 [第21章 项目实战：AI 编程助手](../chapter_coding_agent/README.md)

---

## 本章概览

本章覆盖 Agent 从开发环境到生产环境的完整路径：部署架构设计、API 服务化、Docker 容器化、流式响应与并发处理，最终构建一个完整的生产级 Agent 服务。

## 本章目标

- ✅ 理解生产级 Agent 的分层部署架构
- ✅ 用 FastAPI 封装 Agent 为 API 服务
- ✅ 用 Docker Compose 编排多服务部署
- ✅ 实现流式响应和高并发处理
- ✅ 完成一个生产级 Agent 的端到端部署

## 本章结构

| 小节 | 内容 |
|------|------|
| 18.1 Agent 应用的部署架构 | 分层架构、状态管理 |
| 18.2 API 服务化 | FastAPI 封装、SSE 流式 |
| 18.3 容器化与云部署 | Dockerfile、Docker Compose |
| 18.4 流式响应与并发处理 | 异步、信号量、队列 |
| 18.5 实战：生产级 Agent 服务 | 完整部署流程 |

## ⏱️ 预计学习时间

约 **120-150 分钟**（含部署实践）

## 💡 前置知识

- 已完成第 13-14 章的评估和安全知识
- 了解 HTTP API 和 REST 概念
- 了解 Docker 的基本概念（不要求精通）

## 🔗 学习路径

> **前置知识**：[第18章 评估与优化](../chapter_evaluation/README.md)、[第17章 安全与可靠性](../chapter_security/README.md)
>
> **后续推荐**：
> - 👉 [第19章 AI 编程助手](../chapter_coding_agent/README.md) — 综合项目实战
> - 👉 [第22章 数据分析 Agent](../chapter_data_agent/README.md) — 综合项目实战

---

*下一节：[18.1 Agent 应用的部署架构](./01_deployment_architecture.md)*
