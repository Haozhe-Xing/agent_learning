# 8.7 实战：跨进程恢复长任务

2026 年 11 月公开的 Performance Factory 用独立的任务记录保存性能回归调查阶段、结论和下一步。Agent 提出修复，工程师决定部署内容，部署后再由 Agent 验证恢复情况。这样进程或会话结束后，另一轮执行仍能接手。参见[原始案例](https://openai.com/index/how-openai-performance-factory-keeps-chatgpt-fast/)。

这个离线练习模拟一份报告的发布。SQLite 保留任务状态，命令行操作代表审批，单独的查询验证产物。每条命令都启动一个新进程。

```bash
python examples/chapter08/resumable_task.py start --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db
python examples/chapter08/resumable_task.py approve --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db --stop-after-apply
python examples/chapter08/resumable_task.py run --db demo.db
python examples/chapter08/resumable_task.py run --db demo.db
```

预期状态依次为 `proposed → proposed → approved → applied → verified → verified`。第二条命令验证未审批时不会写入产物。第四条模拟写入提交后、核验前退出。最后两条分别恢复核验与重复执行，数据库中始终只有一份产物。

```python
{{#include ../../../examples/chapter08/resumable_task.py}}
```

示例把产物和检查点放在同一事务。调用外部系统时，持久化操作标识并传递幂等键，重试前检查外部结果。审批也应绑定到具体的产物版本，并由身份受信的服务端执行。延伸阅读 [7.3 长时程任务的上下文策略](../chapter_context_engineering/03_long_horizon.md)和 [13.5 人机协作](../chapter_langgraph/05_human_in_the_loop.md)。
