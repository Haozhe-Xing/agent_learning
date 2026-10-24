# 3.2 Function Calling 机制

本节采用 **Chat Completions的function工具消息形式**。OpenAI SDK本地验证版本为2.54.0；固定提供者不需要SDK或密钥，SDK类型与HTTP替身测试需集成环境。在线文档核对日期为2026-10-22，不能把SDK安装成功等同于真实模型已支持所有设置。

![工具提议、宿主检查与调用ID往返](../svg/chapter_tools_02_function_calling.svg)

## Schema是能力说明，不是执行许可

```python
from reference_code.tool_calling import tool_specs

schema = next(tool for tool in tool_specs() if tool["function"]["name"] == "calculate")
assert schema["type"] == "function"
assert schema["function"]["strict"] is True
assert schema["function"]["parameters"]["required"] == ["expression"]
assert schema["function"]["parameters"]["additionalProperties"] is False
print(schema)
```

完整对象见 `reference_code/tool_calling/core.py:tool_specs`。其中 `strict`、字符串类型、必填字段和禁止额外属性描述可接受的调用形态。按[官方Function Calling说明](https://developers.openai.com/api/docs/guides/function-calling)，严格模式有受支持Schema及配置条件；对象需禁止额外字段，properties里的字段需列入required。本例显式声明模式，不依赖API默认值。

宿主仍需检查JSON、长度、类型、字段、数值/资源范围和当前授权。Schema不能证明数据真实、调用必要或参数对应正确用户；其他提供者、离线替身和保存的消息也不应跳过入口检查。类型注解或dataclass本身不等于运行时验证。

## 一次工具调用必须关联回同一个ID

```python
import json
from reference_code.tool_calling import ToolHost, function_call

call = function_call("calc-demo-1", "calculate", {"expression": "12 * 8"})
assistant = {"role": "assistant", "content": None, "tool_calls": [call]}
result = ToolHost().execute(call["function"]["name"], call["function"]["arguments"])
tool_message = {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}
messages = [{"role": "user", "content": "Calculate 12 * 8"}, assistant, tool_message]
assert messages[-1]["tool_call_id"] == messages[-2]["tool_calls"][0]["id"]
assert json.loads(messages[-1]["content"])["value"]["value"] == 96
print([message["role"] for message in messages])
```

输出角色序列是 `user/assistant/tool`。先保留assistant的调用消息，再为其中每个调用加入同一 `tool_call_id` 的tool消息。不要只把工具结果当普通assistant回答，也不要用新ID替换旧ID。完整循环还会把这些消息交回提供者，再验收最终回答。

ID用于协议关联，不是登录身份、工具许可或跨重试幂等键。配套拒绝空、重复或已使用的ID，且在执行批次前检查整个调用列表；不支持的消息形态明确失败。本例ID限定为1–64位ASCII字母/数字/下划线/连字符，是教学子集的限制。

## SDK类型不能替代业务参数检查

```python
from importlib.metadata import version
from openai.types.chat import ChatCompletionMessage
from reference_code.tool_calling import ToolHost, function_call

assert version("openai") == "2.54.0"
# Deserialize an artificial SDK message, without a client or request.
wire = function_call("sdk-demo-1", "calculate", {"expression": True})
message = ChatCompletionMessage(role="assistant", content=None, tool_calls=[wire])
call = message.tool_calls[0]
assert isinstance(call.function.arguments, str)
result = ToolHost().execute(call.function.name, call.function.arguments)
assert result["code"] == "invalid_arguments"
print(result)
```

这段代码只创建人工SDK消息。`arguments`仍为JSON字符串；本地消息类型不会执行 `calculate` 的业务校验。工具入口拒绝布尔表达式。测试另用真实SDK和 `httpx.MockTransport` 验证请求/响应序列化及三轮ID往返，未请求线上服务、未验证服务端strict或模型选择质量。

## 控制提议与控制执行是两回事

| Chat Completions设置 | 提议层含义 | 宿主仍负责 |
|---|---|---|
| `tool_choice="auto"` | 可直接回答或提出工具 | 无工具时也检查任务是否满足 |
| `"none"` | 本次不请求工具调用 | 保持任务要求，不虚构工具结果 |
| `"required"` | 要求调用工具 | 参数、许可和正确性检查 |
| 指定function对象 | 指定一个工具 | 防止越权和错误参数 |
| `parallel_tool_calls=False` | 限制该配置的同轮工具提议 | 仍校验实际返回的批次；不等于执行超时 |

字段形态以[Chat Completions参考](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)为准；Responses使用不同的调用/返回对象，本节不混用两套消息。允许多个提议也不会自动并发执行，依赖前一个结果的步骤须等结果写回后才能提出。

## 错误、停止与后续协议

配套每个获准进入的调用批次都补齐对应tool结果；一个失败后，其余调用以 `skipped_after_failure` 标记，不再执行。随后任务结束为未完成，不把拒绝变成“请模型换个工具绕过”。若整批超工具预算，则执行前拒绝整批，不把未解决的消息继续发给提供者。

`length`截断、拒答、畸形消息和提供者异常不能当作正常最终答案；空白最终正文同样失败。失败类型与恢复策略继续看[3.3](./03_custom_tools.md)。MCP提供另一层发现/会话/传输约定，不能从服务发现推导执行许可，见[17.5](../chapter_19_protocol/05_practice_mcp_integration.md)。

练习手工构造两个调用：正常ID各有一个结果；再重复ID、漏ID或把结果挂到另一个ID。比较协议验证与函数参数验证分别在哪一层失败，提交消息序列和执行次数。

*下一节：[3.3 自定义工具的设计与实现](./03_custom_tools.md)*
