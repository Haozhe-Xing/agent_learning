# 20.5 Practice: Deploying a Minimal Agent Service

Use the actual `reference-agent/` package to connect the tool loop, fixed evaluation, and basic input checks to FastAPI. This is a local verification baseline. Deployment still requires authentication, rate limits, isolation, logging, and recovery.

```bash
pip install -e "./reference-agent[dev]"
uvicorn reference_agent.service.app:app --host 127.0.0.1 --port 8000
```

The default `FakeProvider` needs no model credential. From another terminal:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"12*8"}'
```

The response should contain 96. Only `AGENT_REAL=1` activates a live model; configure `OPENAI_API_KEY`, optionally `AGENT_MODEL` and `OPENAI_BASE_URL`, without logging credentials.

| Endpoint | Actual behavior | Checks |
|---|---|---|
| `/chat` | Complete execution, then JSON response | Tool-derived output; 400 for rejected input; 500 for execution error |
| `/stream` | Generate the full answer, then split into SSE chunks | Correct content type and event format; explicit execution failure |

The SSE endpoint demonstrates transport, not real-time model-token generation. A live stream also needs model streaming, tool-call events, and cancellation handling.

```bash
cd reference-agent
python -m pytest tests -q
docker build -f docker/Dockerfile -t agent-learning-demo .
docker run --rm -p 127.0.0.1:8000:8000 agent-learning-demo
```

Tests cover offline tools, basic input checks, SSE, JSON-RPC, and fixed evaluation. The packaged task set is `reference-agent/src/reference_agent/data/eval_samples.jsonl`. Its results do not establish live-task quality. SDK tests preserve assistant tool requests and matching `tool_call_id` values.

The Dockerfile installs the package from its own project metadata. Container building and deployment require separate verification; local API tests do not establish production readiness. Recheck identity, tool permissions, budgets, sensitive logs, and recovery when adding business tools.
