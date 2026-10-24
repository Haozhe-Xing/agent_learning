# 数据与完成契约索引

贯穿参考项目的契约已集中到 [reference_code/evidence_assistant/CONTRACTS.md](../reference_code/evidence_assistant/CONTRACTS.md)，包含本地资料调研、六阶段基础练习、来源引用、独立验收和预算停止规则。实现、样例和相关测试也在同一个项目目录。

关联章节仍有独立入口：

- [开发团队实现](dev_team/dev_team.py)及其[反例测试](dev_team/tests/test_dev_team.py)：实际测试、重试、超时与失败不交付。
- [协议与服务参考底座](../reference-agent/README.md)：工具许可、MCP 和 HTTP。

执行结束与验收通过分别记录；只有满足相应项目的验收条件才能交付。各入口的实际验证结果见 [VALIDATION.md](VALIDATION.md)。
