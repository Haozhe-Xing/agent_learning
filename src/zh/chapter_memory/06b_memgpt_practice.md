# 4.7 实战：MemGPT/Letta 记忆架构工程实践

本节用一个可运行的分层记忆基线解释 Core、Working 和 Archive 的职责。它借鉴分层思想，不是 Letta SDK 的替代实现，也不声称达到生产级安全性。完整模块位于 `reference_code/evidence_assistant/memory.py`。

## 三层如何分工

| 层 | 示例中的实现 | 保留边界 |
|---|---|---|
| Core | SQLite 中按用户保存的名称和值 | 显式同意后写入，实例重启后保留 |
| Working | 有容量上限的 `deque` | 只保留本实例近期信息，满时淘汰最早项 |
| Archive | SQLite 中独立的归档记录 | 按查询关键词检索，不把全部历史常驻模型输入 |

自动推断用户偏好是额外的模型能力。本例使用显式的 `update_core_memory` 和 `archive_content`，先验证写入、隔离、检索、删除的工程契约，再接入自动提取。

## 运行完整示例

```bash
python examples/chapter04/layered_memory_agent.py
python examples/chapter04/long_term_memory.py
python examples/chapter04/scratchpad_agent.py
```

```python
from reference_code.evidence_assistant.memory import LayeredMemoryAgent, MemoryStore

store = MemoryStore(":memory:")
agent = LayeredMemoryAgent(store, "demo", working_limit=2)
agent.update_core_memory("name", "Ada", consent=True)
agent.archive_content("project", "churn prediction", consent=True)
messages = agent.messages("churn")
assert "churn prediction" in messages[1]["content"]
agent.forget()
assert agent.search_archive("churn") == {}
store.close()
```

用磁盘文件替代 `:memory:` 可以验证进程或实例重启后的读取。`reference_code/evidence_assistant/tests/test_learning_path.py` 检查磁盘持久化、用户隔离、删除与工作记忆上限。向量检索版的 ChromaDB 验证见 `examples/integration_tests/test_frameworks_and_storage.py`；其中数据库是真实依赖，嵌入向量为固定测试数据。

## 信任与权限

核心记忆也可能保存错误内容或外部指令，不能因为“常驻”就成为系统规则。示例把三层记忆序列化为参考资料，系统消息只写宿主规则。数据分区由用户 ID 限定，但真实服务仍必须先鉴权，不能信任请求方任意填写的身份。

未给 `consent=True` 的写入会被拒绝。`forget` 同时清理该用户的核心、归档和当前实例工作记忆，不影响其他用户。接入生产系统时，应另外定义备份、日志与其他副本的保留和删除策略。

## 何时再接 Letta

先用固定用例明确需要的记忆操作，再按 [Letta 官方文档](https://docs.letta.com/)选择对应接口和版本。不要把本节的独立基线与某个版本的 Letta SDK 混用；框架升级后应重新验证写入权限、召回和删除语义。
