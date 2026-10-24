# 第5章 规划与推理（Planning & Reasoning）

**阅读层次**：核心必读；论文与搜索扩展按需选读。计划、反思和公开解释是可检查的数据，不是私有推理日志或独立事实证明。

## 本章迭代

- **上一版的不足**：第4章保存了偏好与任务状态，但尚未确定步骤依赖、漏目标或失败后应舍弃哪些旧结果。旧的“输出文字/执行完毕”还容易被误判为完成。
- **本章增加**：用“发布简报”小任务练习结构动作、DAG检查、目标映射、反馈修订与总预算。复用既有执行队列，独立检查审批记录、预算和回退项，失败不以自评分放行。
- **怎样验证**：运行 `python -m reference_code.planning_patterns revise`、`python -m reference_code.planning_patterns stubborn`及 `python -m unittest examples.integration_tests.test_planning_chapter -v`。保留计划、公开动作/观察、缺项、预算及最终判定。固定适配器不理解任意自然语言，也没有真实发布动作。
- **下一步**：正确的计划仍可能读取错误资料。到[第6章](../chapter_rag/README.md)加入来源加载、检索与上下文，再分别核对支撑和要点覆盖；多资料、多问题的已组合应用见[贯穿项目](../learning_project.md)。

## 先完成一个有边界的小任务

任务需要审批记录、30 units预算和回退到release-1.0快照三项带来源的简报。`reference_code/planning_patterns/`默认只用标准库、无模型/网络；固定写作者可以漏项或按反馈修订。审批记录只是资料，不授予发布权限。这里没有任意代码、shell或文件写入工具。

```bash
python -m reference_code.planning_patterns plan
python -m reference_code.planning_patterns react
python -m reference_code.planning_patterns revise
python -m reference_code.planning_patterns stubborn
```

前三条默认通过该固定合同；最后一条持续漏项，返回 `budget_exhausted/accepted=false`及退出码1。`trace`记录实际公开动作、观察与错误，`verification`区分缺项、不受来源支持及违反冻结要求；不要把执行结束当业务完成。

已有 `examples/chapter05/plan_and_execute.py`和 `reference_code.evidence_assistant.tests.test_learning_path.PlanningTests`仍保留，适合单独观察失败后队列替换；其 `execution.completed`不是产物验收。本章不修改共享核心，新增练习在外层补足目标与依赖合同。

## 本章结构与学习产物

| 小节 | 本节任务 | 读完提交什么 |
|---|---|---|
| [5.1 Agent 如何“思考”？](./01_how_agents_think.md) | 区分公开计划、观察、假设与证据 | 任务合同和停止理由 |
| [5.2 ReAct：推理 + 行动](./02_react_framework.md) | 结构动作、宿主观察与结束提议 | 正常/非法动作/过早结束轨迹 |
| [5.3 任务分解](./03_task_decomposition.md) | 有界依赖图与声明覆盖 | DAG、目标映射和拒绝反例 |
| [5.4 反思与自我纠错](./04_reflection.md) | 反馈只影响修订，验收保持独立 | 缺项修订与错误自评对照 |
| [5.5 实战：研究助手](./05_practice_research_agent.md) | 另一个较小问答接口练习 | 对照其已配置验证器和引用边界 |
| [5.6 Plan-and-Execute 与 Test-time Compute Scaling](./07_plan_and_execute.md) | 队列替换、总预算与额外计算取舍 | 未通过状态、尝试数和比较记录 |
| [5.7 论文解读](./06_paper_readings.md) | 研究扩展，按需核对原始设置 | 方法与本项目差异，不冒充复现 |

5.5保留的较小API与本章发布简报不是同一个数据集；默认引用标签检查不能代替完整支撑/覆盖验收。需要完整的本地资料应用，可运行 `python -m reference_code.evidence_assistant`，其默认任务另有冻结规格。不要把两个fixture的通过率混为模型表现。

## 前置与停止条件

前置：[第3章工具调用](../chapter_tools/README.md)、[第4章记忆](../chapter_memory/README.md)，以及Python函数与字典。默认练习不需要SDK账号或框架。先保证依赖、来源与独立判据闭环，再按需要使用[第10章](../chapter_langchain/README.md)的组件或[第11章](../chapter_langgraph/README.md)的显式状态/恢复；它们不是本章前置。

只有固定任务的独立要求通过，才能称该任务完成。步数、决策、重规划和字节上限各有单位；它们不会自动中止阻塞回调或撤销副作用。持久恢复、真实模型、并行调度、HTN/MCTS与生产发布均在当前小型配套范围之外。

*下一节：[5.1 Agent 如何“思考”？](./01_how_agents_think.md)*
