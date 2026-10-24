# Chapter 20: Deployment and Production

**Reading level**: Required before deployment.

## This chapter's iteration

- **Previous limitation**: Local execution and acceptance still leave interface errors, process faults, concurrency and resource usage unresolved.
- **This chapter adds**: Define service contracts for results, failures, configuration, logs and recovery. Separate local-function, interface and business-release gates.
- **How to verify**: Supplied `reference-agent/tests/test_service.py` checks the default tool task, rejected input and SSE transport; section 20.5 provides a local service. It is not an HTTP adapter for the Moon-evidence task. Connecting that task, adding authentication/rate limits, cancellation/restart/load checks and container validation are reader deployment exercises. Unverified gates remain open.
- **Next step**: Choose one [capstone](../part6.md), or finish the application route when the current service meets independent acceptance. [Research](../part7.md) is optional.

> Getting code to run locally is just the first step. The real challenge is making Agents serve users reliably in production.

---

## Chapter Overview

This chapter organizes architecture, API, container, streaming and operational concerns around explicit release gates. The supplied local service is a teaching baseline; readers implement and verify the remaining gates for their deployment target.

## Chapter Goals

- ✅ Understand the layered deployment architecture for production-grade Agents
- ✅ Wrap an Agent as an API service using FastAPI
- ✅ Orchestrate multi-service deployment with Docker Compose
- ✅ Implement streaming responses and high-concurrency handling
- ✅ Run the local service baseline and document verified and still-open deployment gates

## Chapter Structure

| Section |
|---|
| [20.1 Deployment Architecture for Agent Applications](./01_deployment_architecture.md) |
| [20.2 API Service: FastAPI / Flask Wrapping](./02_api_service.md) |
| [20.3 Containerization and Cloud Deployment](./03_containerization.md) |
| [20.4 Streaming Responses and Concurrent Processing](./04_streaming_concurrency.md) |
| [20.5 Practice: Deploying a Minimal Agent Service](./05_practice_production_agent.md) |
| [20.6 Model Inference Serving](./06_model_serving.md) |
| [20.7 Kubernetes Orchestration and Serverless GPU](./07_k8s_serverless.md) |
| [20.8 Long-Running Task Queues and Cost Governance](./08_task_queue_cost.md) |

## 🔗 Learning Path

Bring task, permission and acceptance records from [tools](../chapter_tools/README.md) and the [Harness](../chapter_harness/README.md). Production and domain delivery require [evaluation](../chapter_20_evaluation/README.md) and [security](../chapter_21_security/README.md) gates.

Choose one [capstone](../part6.md), or finish the application route when the current service meets independent acceptance. [Research](../part7.md) is optional.

---

*Next: [20.1 Agent Application Deployment Architecture](./01_deployment_architecture.md)*
