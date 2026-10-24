# 第20章 部署与生产化

**阅读层次**：上线必读。

## 本章迭代

- **上一版的不足**：本地执行和验收通过之后，接口错误、进程故障、并发请求与资源消耗仍可能使服务无法稳定交付。
- **本章增加**：为任务结果、失败响应、配置、日志与恢复制定服务契约，区分本地函数可用、接口可用和业务上线三个门禁。
- **怎样验证**：已有 `reference-agent/tests/test_service.py` 检查默认工具任务、输入拒绝和 SSE 传输，20.5 给出本地服务入口。它不是月球资料任务的现成 HTTP 适配器；把贯穿任务接入 API、补鉴权/限流、取消/重启/负载与容器验证是读者部署作业。未验证项应保留为未通过门禁。
- **下一步**：选择[第21–23章的一个领域项目](../part6.md)继续交付；若当前服务已满足独立验收，可以结束应用路线，按需再读[研究进阶](../part7.md)。

> 代码在本地跑通只是第一步。让 Agent 在生产环境稳定服务用户，才是真正的考验。

---

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 掌握 Agent 应用的部署架构设计原则
- ✅ 用 FastAPI 将 Agent 封装为 REST API 服务
- ✅ 实现 Docker 容器化和基本云部署流程
- ✅ 处理流式响应和高并发场景
- ✅ 运行本地教学服务，并为目标环境记录和验证上线门禁

## 📚 本章结构

| 小节 |
|---|
| [20.1 Agent 应用的部署架构](./01_deployment_architecture.md) |
| [20.2 API 服务化：FastAPI / Flask 封装](./02_api_service.md) |
| [20.3 容器化与云部署](./03_containerization.md) |
| [20.4 流式响应与并发处理](./04_streaming_concurrency.md) |
| [20.5 实战：部署 Agent 服务的最小基线](./05_practice_production_agent.md) |
| [20.6 模型推理服务化](./06_model_serving.md) |
| [20.7 Kubernetes 编排与 Serverless GPU](./07_k8s_serverless.md) |
| [20.8 长任务队列与成本治理](./08_task_queue_cost.md) |

## 🔗 学习路径

[工具调用](../chapter_tools/README.md)与[Harness](../chapter_harness/README.md)中的任务、权限和验收记录。生产与领域交付前需通过[评估](../chapter_20_evaluation/README.md)和[安全](../chapter_21_security/README.md)检查。

选择[第21–23章的一个领域项目](../part6.md)继续交付；若当前服务已满足独立验收，可以结束应用路线，按需再读[研究进阶](../part7.md)。

---

*下一节：[20.1 Agent 应用的部署架构](./01_deployment_architecture.md)*
