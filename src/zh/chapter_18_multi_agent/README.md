# 第16章 多 Agent 协作

**阅读层次**：工程进阶；先验证单 Agent 的局限。

## 本章迭代

- **上一版的不足**：单 Agent 的工具、状态与验收已经可控，但研究子任务可能彼此独立；多加角色也可能只增加交接和调用成本。
- **本章增加**：为交接定义输入、产物、权限与总预算，并在汇总前验收子任务。开发团队示例把持续失败、超时和成功交付分开。
- **怎样验证**：已有 `examples/dev_team/tests/test_dev_team.py` 检查实际测试返回码、失败不交付、进程清理和目录隔离。再做读者对照：同一任务集、相同总预算下比较单 Agent 与团队，报告通过率、交接失败与总调用量；团队收益尚未由这些离线用例证明。
- **下一步**：有跨进程工具需求时读[第17章](../chapter_19_protocol/README.md)；没有则进入[第18章评估](../chapter_20_evaluation/README.md)。协作不是上线的前置条件。

> 👥 *"只有分工收益超过交接成本时，协作才值得引入。"*

---

## 🎓 学习目标

完成本章学习后，你将能够：

- ✅ 理解单 Agent 的局限性以及多 Agent 的优势
- ✅ 掌握多 Agent 通信模式（消息队列、共享状态、直接调用）与协商机制（投票、辩论、合同网竞价）
- ✅ 处理共享状态的并发写冲突（reducer / 单写者 / 乐观并发）
- ✅ 设计合理的角色分工、任务分配与负载均衡策略
- ✅ 落地系统级容错方案（超时、有界重试、熔断、心跳、幂等）
- ✅ 区分 Supervisor / Handoff / 去中心化等协作模式并识别六大协作模式的适用场景
- ✅ 使用 LangGraph 构建一个多 Agent 软件开发团队作为综合实战

## 📚 本章结构

| 小节 |
|---|
| [16.1 单 Agent 的局限性](./01_single_agent_limits.md) |
| [16.2 多 Agent 通信模式](./02_communication_patterns.md) |
| [16.3 角色分工与任务分配](./03_role_assignment.md) |
| [16.4 Supervisor 模式 vs. 去中心化模式](./04_supervisor_vs_decentralized.md) |
| [16.5 实战：多 Agent 软件开发团队](./05_practice_dev_team.md) |
| [16.6 论文解读：多 Agent 系统前沿研究](./06_paper_readings.md) |
| [16.7 框架补充：CrewAI（角色扮演型多 Agent 框架）](./07_crewai.md) |

## 🔗 学习路径

[工具调用](../chapter_tools/README.md)与[Harness](../chapter_harness/README.md)中的任务、权限和验收记录。生产与领域交付前需通过[评估](../chapter_20_evaluation/README.md)和[安全](../chapter_21_security/README.md)检查。

有跨进程工具需求时读[第17章](../chapter_19_protocol/README.md)；没有则进入[第18章评估](../chapter_20_evaluation/README.md)。协作不是上线的前置条件。

---

*下一节：[16.1 单 Agent 的局限性](./01_single_agent_limits.md)*
