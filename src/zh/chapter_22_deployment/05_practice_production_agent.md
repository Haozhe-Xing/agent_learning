# 20.5 实战：部署 Agent 服务的最小基线

本节使用实际存在的 `reference-agent/`，把第3章的工具循环、固定评估和基础输入检查接成 FastAPI 服务。它提供本地验证起点；上线前还需鉴权、限流、资源隔离、日志与恢复机制。

## 安装并运行

从仓库根目录运行：

```bash
pip install -e "./reference-agent[dev]"
uvicorn reference_agent.service.app:app --host 127.0.0.1 --port 8000
```

默认使用 `FakeProvider`，无需模型 Key。另开终端：

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"12*8"}'
```

响应应包含 96。只有显式设置 `AGENT_REAL=1` 才连接真实模型；同时配置 `OPENAI_API_KEY`，可按需设置 `AGENT_MODEL` 和 `OPENAI_BASE_URL`。不要在日志或共享命令中写入真实 Key。

## 两个接口的实际行为

| 接口 | 行为 | 应检查的结果 |
|---|---|---|
| `/chat` | 完整执行 Agent 后返回 JSON | 成功输出来自工具；输入拒绝返回 400；执行失败返回 500 |
| `/stream` | 先生成完整答案，再按空白切成 SSE 数据块 | `text/event-stream` 与数据格式正确；执行错误明确返回 |

`/stream` 演示的是 SSE 传输，不能据此声称实现了模型 token 的实时生成。真正的生成流式需要继续接入模型的流式接口、工具调用事件和取消传播，见本章前面的流式专题。

## 固定评估与接口验收

```bash
cd reference-agent
python -m pytest tests -q
```

测试覆盖离线工具结果、输入检查、SSE、JSON-RPC 和固定评估任务。固定任务集随包保存在 `reference-agent/src/reference_agent/data/eval_samples.jsonl`，可用于回归，不代表真实业务已达到相同成功率。模型调用链还需保留 assistant 的工具请求和对应 `tool_call_id`；配套 SDK 测试检查了这一协议约束。

## 容器入口

```bash
cd reference-agent
docker build -f docker/Dockerfile -t agent-learning-demo .
docker run --rm -p 127.0.0.1:8000:8000 agent-learning-demo
```

Dockerfile 从同一份 `pyproject.toml` 安装包，避免只安装依赖却找不到 `reference_agent`。容器构建与部署环境须另外验证；本地接口测试通过不等于完成生产上线验收。

上线前至少复核宿主身份、工具许可、任务预算、敏感数据日志和故障恢复。继续在综合项目中替换业务工具时，重跑相同的成功与失败用例。
