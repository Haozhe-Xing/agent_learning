# 可观测性：日志、追踪与监控

> **本节目标**：学会为 Agent 构建完善的可观测性体系，做到"出了问题能发现、发现了能定位"。

---

## 什么是可观测性？

可观测性（Observability）是指：在不修改系统代码的情况下，通过系统的外部输出来理解其内部状态。对于 Agent 来说，就是能回答以下问题：

- Agent 做了什么决策？为什么？
- 调用了哪些工具？每个工具花了多长时间？
- 用户问题到最终回答之间，经历了哪些中间步骤？
- 出错了，错在哪一步？

可观测性的三大支柱：**日志（Logs）**、**追踪（Traces）**、**指标（Metrics）**。

![可观测性三大支柱](../svg/chapter_evaluation_05_observability.svg)

---

## 支柱一：结构化日志

```python
import logging
import json
from datetime import datetime

class AgentLogger:
    """Agent 专用的结构化日志器"""
    
    def __init__(self, agent_name: str, log_file: str = None):
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(console_handler)
        
        # 文件输出（JSON 格式）
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(file_handler)
    
    def log_event(self, event_type: str, **kwargs):
        """记录结构化事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "event": event_type,
            **kwargs
        }
        self.logger.info(json.dumps(event, ensure_ascii=False))
    
    def log_llm_call(
        self,
        model: str,
        prompt: str,
        response: str,
        tokens: dict,
        latency: float
    ):
        """记录 LLM 调用"""
        self.log_event(
            "llm_call",
            model=model,
            prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            response_preview=response[:200] + "..." if len(response) > 200 else response,
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
            latency_ms=round(latency * 1000)
        )
    
    def log_tool_call(
        self,
        tool_name: str,
        args: dict,
        result: str,
        success: bool,
        latency: float
    ):
        """记录工具调用"""
        self.log_event(
            "tool_call",
            tool=tool_name,
            arguments=args,
            result_preview=str(result)[:200],
            success=success,
            latency_ms=round(latency * 1000)
        )
    
    def log_error(self, error: Exception, context: dict = None):
        """记录错误"""
        self.log_event(
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )

# 使用示例
logger = AgentLogger("customer_service", log_file="agent.log")

logger.log_llm_call(
    model="gpt-4o",
    prompt="用户问：我的订单到哪了？",
    response="让我帮您查询一下订单状态...",
    tokens={"input": 150, "output": 80},
    latency=1.2
)
```

---

## 支柱二：链路追踪

追踪一个请求从开始到结束经历的所有步骤：

```python
import uuid
import time
from dataclasses import dataclass, field

@dataclass
class Span:
    """追踪链路中的一个节点"""
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_id: str = None
    start_time: float = 0.0
    end_time: float = 0.0
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    
    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


class AgentTracer:
    """Agent 链路追踪器"""
    
    def __init__(self):
        self.traces = {}  # trace_id -> list[Span]
    
    def start_trace(self, name: str) -> Span:
        """开始一条新的追踪链路"""
        trace_id = str(uuid.uuid4())[:12]
        span = Span(name=name, trace_id=trace_id)
        span.start_time = time.time()
        self.traces[trace_id] = [span]
        return span
    
    def start_span(self, name: str, parent: Span) -> Span:
        """在现有链路中创建子节点"""
        span = Span(
            name=name,
            trace_id=parent.trace_id,
            parent_id=parent.span_id
        )
        span.start_time = time.time()
        self.traces[parent.trace_id].append(span)
        return span
    
    def end_span(self, span: Span, status: str = "ok", **attributes):
        """结束一个节点"""
        span.end_time = time.time()
        span.status = status
        span.attributes.update(attributes)
    
    def print_trace(self, trace_id: str):
        """可视化打印一条完整的追踪链路"""
        spans = self.traces.get(trace_id, [])
        if not spans:
            print("未找到该追踪链路")
            return
        
        print(f"\n{'='*60}")
        print(f"🔍 Trace: {trace_id}")
        print(f"{'='*60}")
        
        # 构建树结构
        root_spans = [s for s in spans if s.parent_id is None]
        
        for root in root_spans:
            self._print_span_tree(root, spans, indent=0)
    
    def _print_span_tree(self, span: Span, all_spans: list, indent: int):
        """递归打印 Span 树"""
        prefix = "  " * indent
        status_icon = "✅" if span.status == "ok" else "❌"
        
        print(f"{prefix}{status_icon} {span.name} "
              f"({span.duration_ms:.0f}ms)")
        
        for key, value in span.attributes.items():
            print(f"{prefix}   {key}: {value}")
        
        # 打印子节点
        children = [s for s in all_spans if s.parent_id == span.span_id]
        for child in children:
            self._print_span_tree(child, all_spans, indent + 1)


# 使用示例
tracer = AgentTracer()

# 模拟一个完整的 Agent 请求追踪
root = tracer.start_trace("handle_user_query")

# 第 1 步：理解用户意图
intent_span = tracer.start_span("classify_intent", root)
# ... 执行意图分类 ...
tracer.end_span(intent_span, intent="order_query")

# 第 2 步：调用工具
tool_span = tracer.start_span("call_tool:query_order", root)
# ... 查询订单 ...
tracer.end_span(tool_span, order_id="12345", status="shipped")

# 第 3 步：生成回复
reply_span = tracer.start_span("generate_reply", root)
# ... 生成最终回复 ...
tracer.end_span(reply_span, tokens=150)

tracer.end_span(root)
tracer.print_trace(root.trace_id)
```

输出示例：
```
============================================================
🔍 Trace: a1b2c3d4e5f6
============================================================
✅ handle_user_query (1523ms)
  ✅ classify_intent (245ms)
     intent: order_query
  ✅ call_tool:query_order (1050ms)
     order_id: 12345
     status: shipped
  ✅ generate_reply (228ms)
     tokens: 150
```

---

## 支柱三：监控指标

```python
import time
from collections import defaultdict, deque
from dataclasses import dataclass

class AgentMonitor:
    """Agent 运行时监控"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.error_count = 0
        self.total_count = 0
        self.tool_stats = defaultdict(
            lambda: {"calls": 0, "errors": 0, "total_ms": 0}
        )
    
    def record_request(self, latency: float, success: bool):
        """记录一次请求"""
        self.total_count += 1
        self.latencies.append(latency)
        if not success:
            self.error_count += 1
    
    def record_tool_usage(
        self,
        tool_name: str,
        latency: float,
        success: bool
    ):
        """记录工具使用情况"""
        stats = self.tool_stats[tool_name]
        stats["calls"] += 1
        stats["total_ms"] += latency * 1000
        if not success:
            stats["errors"] += 1
    
    def get_dashboard(self) -> str:
        """获取监控面板数据"""
        avg_latency = (
            sum(self.latencies) / len(self.latencies)
            if self.latencies else 0
        )
        error_rate = (
            self.error_count / self.total_count
            if self.total_count else 0
        )
        p95_latency = (
            sorted(self.latencies)[int(len(self.latencies) * 0.95)]
            if len(self.latencies) > 20 else avg_latency
        )
        
        dashboard = f"""
┌──────────────────────────────────────┐
│        🖥️  Agent 监控面板             │
├──────────────────────────────────────┤
│ 总请求数:    {self.total_count:<20} │
│ 错误率:      {error_rate:<20.2%} │
│ 平均延迟:    {avg_latency:<18.0f}ms │
│ P95 延迟:    {p95_latency:<18.0f}ms │
├──────────────────────────────────────┤
│ 🔧 工具使用统计                       │
"""
        for name, stats in self.tool_stats.items():
            avg_tool_ms = (
                stats["total_ms"] / stats["calls"]
                if stats["calls"] else 0
            )
            dashboard += (
                f"│ {name:<15} "
                f"调用:{stats['calls']:<5} "
                f"均耗时:{avg_tool_ms:.0f}ms │\n"
            )
        
        dashboard += "└──────────────────────────────────────┘"
        return dashboard
```

---

## 使用 LangSmith 进行追踪（推荐）

[LangSmith](https://smith.langchain.com/) 是 LangChain 官方的可观测性平台，可以自动追踪 LangChain/LangGraph 应用的每一步：

```python
import os

# 只需设置环境变量即可启用 LangSmith 追踪
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# 之后所有 LangChain 调用都会自动被追踪
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
response = llm.invoke("你好")
# 这次调用的详细信息（输入、输出、延迟、Token）
# 会自动出现在 LangSmith 的 Web 界面上
```

LangSmith 提供的核心功能：

| 功能 | 说明 |
|------|------|
| 自动追踪 | 每次 LLM/工具调用的完整链路 |
| 可视化 | 在 Web 界面上查看每步的输入输出 |
| 数据集管理 | 创建测试数据集，批量评估 |
| 比较运行 | 对比不同版本的表现差异 |
| 告警 | 设置错误率、延迟等告警规则 |

---

## 小结

| 支柱 | 解决的问题 | 工具 |
|------|-----------|------|
| 日志 | "发生了什么？" | 结构化日志、JSON 格式 |
| 追踪 | "经过了哪些步骤？" | Span 链路、LangSmith |
| 指标 | "整体表现如何？" | 计数器、直方图、监控面板 |

> 🎓 **本章总结**：评估和优化是一个持续迭代的过程。先建立评估体系，然后通过 Prompt 调优、成本控制和可观测性来不断改进 Agent。

---

[下一章：第17章 安全与可靠性 →](../chapter_security/README.md)
