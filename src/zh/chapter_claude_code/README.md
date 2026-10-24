# 第14章 Claude Code 深度解析

**阅读层次**：编程产品与架构案例选读；按代码任务选择。

本章以 Claude Code 为编程助理案例，研究文件操作、命令执行、上下文和权限怎样影响一个代码任务的结果。架构讨论采用教学视角，区分公开可观察行为、明确版本的实现材料与作者推断。

## 本章迭代

- **上一版的不足**：已有工具与 Harness 约束尚未通过代码修改任务检查：补丁是否符合需求、测试是否真的运行、失败后是否停止？
- **本章增加**：以编程产品案例组织代码操作、上下文、权限与扩展机制的观察方法。
- **怎样验证**：本书没有统一产品适配器。读者应固定仓库、任务、版本和权限，独立检查补丁与测试结果，保留授权和失败记录；无法运行时可审读设计，但须标明“未运行”。
- **下一步**：研究组件替换时选读[第15章 DeepSeek Harness](../chapter_deepseek_harness/README.md)；需要自己实现编程助手时进入[第21章 编程助手项目](../chapter_23_coding_agent/README.md)。

## 本章结构

| 小节 | 本节关注的问题 |
|---|---|
| [14.1 工业级 Harness 的前世：从 AutoGPT 到 Claude Code](./01_industry_history.md) | 理解代码助理与执行闭环的任务背景 |
| [14.2 认识 Claude Code：从零到上手](./02_introduction.md) | 记录交互方式与实验配置 |
| [14.3 核心架构深度解析](./03_architecture.md) | 用本书分析模型梳理工具与状态 |
| [14.4 System Prompt、权限工程与 Prompt Cache](./04_system_prompt_and_permissions.md) | 提示、权限、上下文与缓存边界 |
| [14.5 高级用法：MCP、Hooks 与 Skills](./05_advanced_usage.md) | 检查扩展接口的授权与执行结果 |
| [14.6 生产实践与团队配置](./06_production_and_team.md) | 团队配置、失败记录与验收要求 |

## 阅读路径

理解[第3章 工具调用](../chapter_tools/README.md)、[第7章 上下文](../chapter_context_engineering/README.md)和[第8章 Harness](../chapter_harness/README.md)即可开始对应小节；第12–13章产品案例不是前置。使用目标从 14.2 进入，架构研究可选择 14.3–14.4。

## 验证与材料边界

本章不以许可、可逐行阅读内部源码、固定提示行数或工具数量作为选型理由。若某项结构来自社区分析，应给出材料与版本并标记推断；可验证的行为仍需读者在自己的实验环境复现。

仓库的开发团队与编程底座示例验证自身代码路径，不能直接证明 Claude Code 的实现或真实模型质量。对生成代码的检查应使用独立验收，不能把产品结束提示等同于交付成功。

---

*下一节：[14.1 工业级 Harness 的前世：从 AutoGPT 到 Claude Code](./01_industry_history.md)*
