# 13.4 Dify / Coze 等低代码 Agent 平台

低代码平台让非程序员也能构建 Agent 应用，降低了 AI 应用的开发门槛。2024-2025 年，这一领域涌现了大量平台，竞争日趋激烈。

## 主流低代码平台

### Dify

Dify 是目前最流行的开源 LLM 应用开发平台，截至 2026 年初已在 GitHub 获得 90K+ Star：

![Dify 主要特性](../svg/chapter_frameworks_04_dify_features.svg)

Dify 通过 API 接入现有应用：

```python
import requests

# Dify 应用 API 调用示例
def call_dify_app(app_token: str, user_message: str) -> str:
    """调用 Dify 应用"""
    url = "https://api.dify.ai/v1/chat-messages"
    
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {app_token}",
            "Content-Type": "application/json"
        },
        json={
            "inputs": {},
            "query": user_message,
            "response_mode": "blocking",
            "user": "user_001"
        }
    )
    
    result = response.json()
    return result.get("answer", "")

# 使用示例
answer = call_dify_app("your-app-token", "如何申请报销？")
print(answer)
```

### Coze（扣子）

字节跳动推出的 Agent 构建平台，是国内低代码 Agent 平台中功能最丰富的之一：

**主要特性**：
- 图形化 Agent 构建，拖拽式工作流编排（支持条件分支、循环）
- 丰富的内置插件（天气、搜索、代码执行、数据库等）
- 多平台发布：微信、飞书、抖音、Discord、Telegram 等
- Bot 市场（可分享和复用）
- 知识库管理（支持 RAG 检索增强）

Coze 同样提供了 API，可以通过代码调用在 Coze 上构建的 Bot：

```python
import requests
import json

def call_coze_bot(
    bot_id: str,
    user_message: str,
    access_token: str,
    user_id: str = "user_001",
) -> str:
    """调用 Coze Bot API"""
    url = "https://api.coze.cn/v3/chat"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "bot_id": bot_id,
            "user_id": user_id,
            "stream": False,
            "auto_save_history": True,
            "additional_messages": [
                {
                    "role": "user",
                    "content": user_message,
                    "content_type": "text",
                }
            ],
        },
    )

    result = response.json()
    # Coze API 返回的消息在 data.messages 中
    if result.get("code") == 0:
        messages = result.get("data", {}).get("messages", [])
        # 找到 assistant 的回复
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("type") == "answer":
                return msg.get("content", "")
    return f"API 调用失败: {result.get('msg', '未知错误')}"


# 使用示例
answer = call_coze_bot(
    bot_id="your-bot-id",
    user_message="帮我分析一下最近的销售数据趋势",
    access_token="your-access-token",
)
print(answer)
```

> 💡 **Coze vs Dify 选择**：Coze 更适合需要快速分发到多个即时通讯平台（微信、飞书、抖音）的场景；Dify 更适合需要私有化部署、自定义工作流的企业场景。

### Dify 工作流 API 调用

除了基础的对话 API，Dify 还支持通过 API 调用**工作流（Workflow）**，这对于将 Dify 编排的复杂逻辑嵌入到自有系统中非常有用：

```python
import requests

def run_dify_workflow(
    api_key: str,
    inputs: dict,
    user: str = "user_001",
) -> dict:
    """调用 Dify Workflow API"""
    url = "https://api.dify.ai/v1/workflows/run"

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "inputs": inputs,
            "response_mode": "blocking",
            "user": user,
        },
    )

    result = response.json()
    return result.get("data", {}).get("outputs", {})


# 示例：调用一个"文档摘要+翻译"工作流
outputs = run_dify_workflow(
    api_key="app-your-workflow-key",
    inputs={
        "document_url": "https://example.com/report.pdf",
        "target_language": "English",
    },
)
print(outputs)
# 输出：{"summary": "...", "translation": "..."}
```

### n8n：工作流自动化平台

n8n 是一个开源的工作流自动化平台，内置了 AI Agent 节点，可以通过拖拽配置实现复杂的 LLM 工作流。它的核心优势是**连接一切** —— 支持 400+ 应用集成。

n8n 通过 Webhook 与外部系统双向集成：

```python
import requests

# 调用 n8n 的 Webhook 触发一个 AI 工作流
# 在 n8n 中配置：Webhook → AI Agent → Slack 通知
def trigger_n8n_workflow(
    webhook_url: str,
    payload: dict,
) -> dict:
    """触发 n8n 工作流"""
    response = requests.post(webhook_url, json=payload)
    return response.json()


# 示例：触发一个"每日任务智能排序"工作流
result = trigger_n8n_workflow(
    webhook_url="https://your-n8n.example.com/webhook/daily-tasks",
    payload={
        "tasks": [
            "完成项目方案",
            "修复线上 bug",
            "写周报",
            "团队代码 review",
        ],
        "context": "下午有一个重要的客户演示",
    },
)
# n8n 工作流：读取任务 → 调用 GPT-4o 分析优先级 → 返回排序结果
print(result)  # {"prioritized_tasks": [...], "reasoning": "..."}
```

### 其他值得关注的平台

| 平台 | 特点 | 适合场景 |
|------|------|---------|
| **FastGPT** | 开源，知识库 RAG 优秀 | 企业知识库问答 |
| **Langflow** | LangChain 可视化编排 | 开发者快速原型 |
| **Flowise** | 低代码 LangChain 编排 | 简单 Agent 搭建 |
| **百度 AppBuilder** | 百度生态，文心模型 | 国内企业应用 |
| **阿里百炼** | 阿里云生态，通义模型 | 国内企业应用 |

## 低代码 vs 代码开发

![低代码vs代码开发对比](../svg/chapter_frameworks_04_lowcode_vs_code.svg)

### 如何选择？

```python
decision_guide = {
    "选低代码平台": [
        "快速验证 MVP（1-3 天内出原型）",
        "非技术团队主导的项目",
        "标准化的客服/问答/文档处理场景",
        "对定制化要求不高的内部工具",
    ],
    "选代码开发": [
        "需要深度定制 Agent 行为逻辑",
        "对延迟/成本/安全有严格要求",
        "需要与已有系统深度集成",
        "多 Agent 协作等复杂架构",
    ],
    "混合方案": [
        "用 Dify 快速搭建原型，验证产品方向",
        "验证可行后用 LangChain/LangGraph 重写核心逻辑",
        "保留 Dify 作为非核心模块的编排工具",
    ],
}
```

---

## 小结

低代码平台降低了 Agent 开发门槛，但代码开发提供了更大的灵活性。最佳实践是：用低代码快速验证想法，用代码做精细实现和生产部署。随着 Dify、Coze 等平台的持续迭代，低代码平台的能力边界正在不断扩大。

---

*下一节：[13.5 如何选择合适的框架？](./05_how_to_choose.md)*
