# 第3章 工具调用（Tool Use / Function Calling）

**阅读层次**：核心必读。

## 本章迭代

- **上一版的不足**：模型可以解释光行时间，却可能混淆单位、只声称调用工具，或在工具报错后仍给出成功答案。
- **本章增加**：从工具Schema到宿主参数/许可检查、实际执行、调用ID往返和独立验收，构建一个可观察的本地搜索与计算循环。
- **怎样验证**：运行 `python -m reference_code.tool_calling` 与 `python -m unittest examples.integration_tests.test_tool_calling_chapter -v`；成功需3轮固定提供者、2次真实工具处理及冻结判据通过。再限制预算、拒绝许可、篡改结果，确认仍为未完成。主线另有 `python -m reference_code.evidence_assistant.demo --stage tools` 的资料/单位合同练习。
- **下一步**：在[第4章](../chapter_memory/README.md)保存用户明确同意保留的单位偏好；计算正确不意味着下次会话仍能使用正确偏好。

工具调用把模型的提议交给宿主程序处理。模型可以选择工具与填写参数，但应用仍决定哪些动作获准、怎样校验、执行失败如何表示、什么证据足以完成任务。工具不是模型获得任意系统权限的入口。

## 学习目标与范围

本章用中等难度的本地任务连起协议和工程边界：检索一条冻结的月球距离教学资料，使用受限算术解释器计算秒数，再独立核对证据、数值、单位与来源。固定提供者消费实际工具结果，不伪装成具备自然语言决策能力的模型。

配套在 `reference_code/tool_calling/`，从仓库根目录运行；复用 `examples/chapter03/safe_math.py` 和主线 `rag.py`，不是可脱离这些模块单目录复制的完整产品。默认仅标准库，不搜索网络、不读任意文件、不调用线上模型。SDK练习用锁定的 OpenAI 2.54.0 类型与HTTP替身，另需仓库集成环境。

| 小节 | 本次增量 | 读完应交付 |
|---|---|---|
| [3.1 为什么需要工具](./01_why_tools.md) | 区分提议、观察和验收 | 工具边界表 |
| [3.2 Function Calling](./02_function_calling.md) | Schema与调用ID消息往返 | 一次关联正确的消息轨迹 |
| [3.3 自定义工具](./03_custom_tools.md) | 参数、许可、返回和预算 | 拒绝发生在执行前的反例 |
| [3.4 工具描述](./04_tool_description.md) | 将文字说明对齐真实合同 | 工具选择的独立用例集 |
| [3.5 搜索与计算实战](./05_practice_search_calc.md) | 依赖步骤、失败停止、独立判据 | 正常和未完成报告 |
| [3.6 论文阅读](./06_paper_readings.md) | 选读研究路线 | 对照论文任务与本地机制的差异 |

先完成离线正常路径和失败用例，再决定是否需要真实模型、网络搜索或更多工具。真实模型的选择质量、外部数据真实性和生产服务能力都不是固定夹具测试的结论。

前置：[第2章](../chapter_llm/README.md)与[环境准备](../chapter_setup/README.md)，会写Python函数和JSON即可。协议标准化继续看[第17章](../chapter_19_protocol/README.md)；访问控制和执行隔离继续看[第19章](../chapter_21_security/README.md)。

*下一节：[3.1 为什么 Agent 需要工具？](./01_why_tools.md)*
