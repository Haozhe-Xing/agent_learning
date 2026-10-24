# Chapter 22: Deployment and Production

> Getting code to run locally is just the first step. The real challenge is making Agents serve users reliably in production.

---

## Chapter Overview

This chapter covers the complete path from development to production for Agents: deployment architecture design, API service wrapping, Docker containerization, streaming responses and concurrency handling, and ultimately building a complete production-grade Agent service.

## Chapter Goals

- ✅ Understand the layered deployment architecture for production-grade Agents
- ✅ Wrap an Agent as an API service using FastAPI
- ✅ Orchestrate multi-service deployment with Docker Compose
- ✅ Implement streaming responses and high-concurrency handling
- ✅ Complete an end-to-end deployment of a production-grade Agent

## Chapter Structure

| Section | Content |
|---------|---------|
| 18.1 Agent Application Deployment Architecture | Layered architecture, state management |
| 18.2 API Service Wrapping | FastAPI encapsulation, SSE streaming |
| 18.3 Containerization and Cloud Deployment | Dockerfile, Docker Compose |
| 18.4 Streaming Responses and Concurrency | Async, semaphores, queues |
| 18.5 Practice: Production-Grade Agent Service | Complete deployment workflow |

## ⏱️ Estimated Study Time

Approximately **120–150 minutes** (including deployment practice)

## 💡 Prerequisites

- Completed Chapters 13–14 on evaluation and security
- Familiarity with HTTP APIs and REST concepts
- Basic understanding of Docker (expertise not required)

## 🔗 Learning Path

> **Prerequisites**: [Chapter 18: Evaluation and Optimization](../chapter_20_evaluation/README.md), [Chapter 19: Security and Reliability](../chapter_21_security/README.md)
>
> **Recommended Next Steps**:
> - 👉 [Chapter 21: AI Coding Assistant](../chapter_23_coding_agent/README.md) — Comprehensive project practice
> - 👉 [Chapter 22: Data Analysis Agent](../chapter_24_data_agent/README.md) — Comprehensive project practice

---

*Next: [22.1 Agent Application Deployment Architecture](./01_deployment_architecture.md)*
