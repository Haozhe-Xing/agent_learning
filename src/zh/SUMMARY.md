# 从零开始学 Agent 开发

[前言](./preface.md)

---

- [第一部分：入门篇](./part1.md)

- [第1章 什么是 Agent？](./chapter_intro/README.md)
  - [1.1 从聊天机器人到智能体的演进](./chapter_intro/01_evolution.md)
  - [1.2 Agent 的核心概念与定义](./chapter_intro/02_core_concepts.md)
  - [1.3 Agent 架构：感知-思考-行动循环](./chapter_intro/03_architecture.md)
  - [1.4 Agent 与传统程序的区别](./chapter_intro/04_agent_vs_traditional.md)
  - [1.5 Agent 的应用场景全景图](./chapter_intro/05_use_cases.md)
  - [1.6 智能体发展史：从符号主义到大模型驱动](./chapter_intro/06_history.md)

- [第2章 开发环境搭建](./chapter_setup/README.md)
  - [2.1 Python 环境与依赖管理](./chapter_setup/01_python_setup.md)
  - [2.2 关键库安装](./chapter_setup/02_install_libs.md)
  - [2.3 API Key 管理与安全最佳实践](./chapter_setup/03_api_key_management.md)
  - [2.4 第一个 Agent：Hello Agent！](./chapter_setup/04_hello_agent.md)

- [第3章 大语言模型基础](./chapter_llm/README.md)
  - [3.1 LLM 是如何工作的？（直觉理解）](./chapter_llm/01_how_llm_works.md)
  - [3.2 Prompt Engineering](./chapter_llm/02_prompt_engineering.md)
  - [3.3 Few-shot / Zero-shot / Chain-of-Thought 提示策略](./chapter_llm/03_prompting_strategies.md)
  - [3.4 模型 API 调用入门](./chapter_llm/04_api_basics.md)
  - [3.5 Token、Temperature 与模型参数详解](./chapter_llm/05_model_parameters.md)
  - [3.6 基座模型前沿进展与选型指南](./chapter_llm/06_foundation_model_landscape.md)
  - [3.7 基座模型架构详解](./chapter_llm/07_model_architecture.md)
  - [3.8 SFT 与强化学习训练数据准备](./chapter_llm/08_training_data.md)

---

- [第二部分：核心能力篇](./part2.md)

- [第4章 工具调用（Tool Use / Function Calling）](./chapter_tools/README.md)
  - [4.1 为什么 Agent 需要工具？](./chapter_tools/01_why_tools.md)
  - [4.2 Function Calling 机制](./chapter_tools/02_function_calling.md)
  - [4.3 自定义工具的设计与实现](./chapter_tools/03_custom_tools.md)
  - [4.4 工具描述的编写技巧](./chapter_tools/04_tool_description.md)
  - [4.5 实战：搜索引擎 + 计算器 Agent](./chapter_tools/05_practice_search_calc.md)
  - [4.6 论文解读：工具学习前沿进展](./chapter_tools/06_paper_readings.md)

- [第5章 记忆系统（Memory）](./chapter_memory/README.md)
  - [5.1 为什么 Agent 需要记忆？](./chapter_memory/01_why_memory.md)
  - [5.2 短期记忆：对话历史管理](./chapter_memory/02_short_term_memory.md)
  - [5.3 长期记忆：向量数据库与检索](./chapter_memory/03_long_term_memory.md)
  - [5.4 工作记忆：Scratchpad 模式](./chapter_memory/04_working_memory.md)
  - [5.5 实战：带记忆的个人助理 Agent](./chapter_memory/05_practice_memory_agent.md)
  - [5.6 论文解读：记忆系统前沿进展](./chapter_memory/06_paper_readings.md)

- [第6章 规划与推理（Planning & Reasoning）](./chapter_planning/README.md)
  - [6.1 Agent 如何"思考"？](./chapter_planning/01_how_agents_think.md)
  - [6.2 ReAct：推理 + 行动框架](./chapter_planning/02_react_framework.md)
  - [6.3 任务分解：将复杂问题拆解为子任务](./chapter_planning/03_task_decomposition.md)
  - [6.4 反思与自我纠错机制](./chapter_planning/04_reflection.md)
  - [6.5 实战：自动化研究助手 Agent](./chapter_planning/05_practice_research_agent.md)
  - [6.6 论文解读：规划与推理前沿研究](./chapter_planning/06_paper_readings.md)

- [第7章 检索增强生成（RAG）](./chapter_rag/README.md)
  - [7.1 RAG 的概念与工作原理](./chapter_rag/01_rag_concepts.md)
  - [7.2 文档加载与文本分割](./chapter_rag/02_document_loading.md)
  - [7.3 向量嵌入与向量数据库](./chapter_rag/03_embeddings_vectordb.md)
  - [7.4 检索策略与重排序](./chapter_rag/04_retrieval_strategies.md)
  - [7.5 实战：智能文档问答 Agent](./chapter_rag/05_practice_qa_agent.md)
  - [7.6 论文解读：RAG 前沿进展](./chapter_rag/06_paper_readings.md)

- [第8章 上下文工程](./chapter_context_engineering/README.md)
  - [8.1 从提示工程到上下文工程](./chapter_context_engineering/01_context_vs_prompt.md)
  - [8.2 上下文窗口管理与注意力预算](./chapter_context_engineering/02_context_window.md)
  - [8.3 长时程任务的上下文策略](./chapter_context_engineering/03_long_horizon.md)
  - [8.4 实战：构建上下文管理器](./chapter_context_engineering/04_practice_context_builder.md)
  - [8.5 上下文工程前沿进展](./chapter_context_engineering/05_latest_advances.md)

- [第9章 Harness Engineering：驾驭 Agent 的系统工程](./chapter_harness/README.md)
  - [9.1 什么是 Harness Engineering？](./chapter_harness/01_what_is_harness.md)
  - [9.2 六大工程支柱](./chapter_harness/02_six_pillars.md)
  - [9.3 AGENTS.md / CLAUDE.md：Agent 宪法写作指南](./chapter_harness/03_agents_md.md)
  - [9.4 生产级案例：OpenAI、LangChain、Stripe](./chapter_harness/04_production_cases.md)
  - [9.5 实战：构建你的第一个 Harness 系统](./chapter_harness/05_practice_harness_builder.md)
  - [9.6 结构化输出：保障 JSON 可靠性的工程实践](./chapter_harness/06_structured_output.md)

- [第10章 Skill System](./chapter_skill/README.md)
  - [10.1 技能系统概述](./chapter_skill/01_skill_overview.md)
  - [10.2 技能的定义与封装](./chapter_skill/02_skill_definition.md)
  - [10.3 技能学习与获取](./chapter_skill/03_skill_learning.md)
  - [10.4 技能发现与注册](./chapter_skill/04_skill_discovery.md)
  - [10.5 实战：构建可复用的技能系统](./chapter_skill/05_practice_skill_system.md)
  - [10.6 论文解读：技能系统前沿研究](./chapter_skill/06_paper_readings.md)
  - [10.7 Tool、Skill 与 Sub Agent：三层能力抽象](./chapter_skill/07_tool_skill_subagent.md)
  - [10.8 Skills圣经：Superpowers工程实践指南](./chapter_skill/08_superpowers_guide.md)

- [第11章 Agentic-RL：智能体强化学习训练](./chapter_agentic_rl/README.md)
  - [11.1 什么是 Agentic-RL](./chapter_agentic_rl/01_agentic_rl_overview.md)
  - [11.2 SFT + LoRA 基础训练](./chapter_agentic_rl/02_sft_lora.md)
  - [11.2b 分布式训练基础：DP / TP / PP / SP / ZeRO](./chapter_agentic_rl/02b_distributed_training.md)
  - [11.3 PPO：近端策略优化](./chapter_agentic_rl/03_ppo.md)
  - [11.4 DPO：直接偏好优化](./chapter_agentic_rl/04_dpo.md)
  - [11.5 GRPO/GSPO：组内相对策略优化与奖励函数设计](./chapter_agentic_rl/05_grpo.md)
  - [11.6 实战：完整 SFT + GRPO 训练 Pipeline](./chapter_agentic_rl/06_practice_training.md)
  - [11.7 最新研究进展（2025—2026）](./chapter_agentic_rl/07_latest_research.md)
  - [11.8 专为 Agent 的微调：让模型真正会用工具](./chapter_agentic_rl/08_agent_finetuning.md)
  - [11.9 Agentic 数据飞轮：让 Agent 自我进化](./chapter_agentic_rl/09_data_flywheel.md)

---

- [第三部分：框架实战篇](./part3.md)

- [第12章 LangChain 深入实战](./chapter_langchain/README.md)
  - [12.1 LangChain 架构全景](./chapter_langchain/01_langchain_overview.md)
  - [12.2 Chain：构建处理管道](./chapter_langchain/02_chains.md)
  - [12.3 使用 LangChain 构建 Agent](./chapter_langchain/03_langchain_agents.md)
  - [12.4 LCEL：LangChain 表达式语言](./chapter_langchain/04_lcel.md)
  - [12.5 实战：多功能客服 Agent](./chapter_langchain/05_practice_customer_service.md)

- [第13章 LangGraph：构建有状态的 Agent](./chapter_langgraph/README.md)
  - [13.1 为什么需要图结构？](./chapter_langgraph/01_why_graph.md)
  - [13.2 LangGraph 核心概念：节点、边、状态](./chapter_langgraph/02_core_concepts.md)
  - [13.3 构建你的第一个 Graph Agent](./chapter_langgraph/03_first_graph_agent.md)
  - [13.4 条件路由与循环控制](./chapter_langgraph/04_conditional_routing.md)
  - [13.5 Human-in-the-Loop：人机协作](./chapter_langgraph/05_human_in_the_loop.md)
  - [13.6 实战：工作流自动化 Agent](./chapter_langgraph/06_practice_workflow_agent.md)

- [第14章 其他主流框架概览](./chapter_frameworks/README.md)
  - [14.1 AutoGPT 与 BabyAGI 的启示](./chapter_frameworks/01_autogpt_babyagi.md)
  - [14.2 CrewAI：角色扮演型多 Agent 框架](./chapter_frameworks/02_crewai.md)
  - [14.3 AutoGen：多 Agent 对话框架](./chapter_frameworks/03_autogen.md)
  - [14.4 Dify / Coze 等低代码 Agent 平台](./chapter_frameworks/04_low_code_platforms.md)
  - [14.5 如何选择合适的框架？](./chapter_frameworks/05_how_to_choose.md)

- [第15章 Claude Code 深度解析：从使用到源码](./chapter_claude_code/README.md)
  - [15.1 认识 Claude Code：从零到上手](./chapter_claude_code/01_introduction.md)
  - [15.2 核心架构深度解析](./chapter_claude_code/02_architecture.md)
  - [15.3 源码解密：System Prompt 与权限工程](./chapter_claude_code/03_source_code_analysis.md)
  - [15.4 高级用法：MCP、Hooks 与 Skills](./chapter_claude_code/04_advanced_usage.md)
  - [15.5 生产实践：在团队中用好 Claude Code](./chapter_claude_code/05_best_practices.md)

---

- [第四部分：多 Agent 系统篇](./part4.md)

- [第16章 多 Agent 协作](./chapter_multi_agent/README.md)
  - [16.1 单 Agent 的局限性](./chapter_multi_agent/01_single_agent_limits.md)
  - [16.2 多 Agent 通信模式](./chapter_multi_agent/02_communication_patterns.md)
  - [16.3 角色分工与任务分配](./chapter_multi_agent/03_role_assignment.md)
  - [16.4 Supervisor 模式 vs. 去中心化模式](./chapter_multi_agent/04_supervisor_vs_decentralized.md)
  - [16.5 实战：多 Agent 软件开发团队](./chapter_multi_agent/05_practice_dev_team.md)
  - [16.6 论文解读：多 Agent 系统前沿研究](./chapter_multi_agent/06_paper_readings.md)

- [第17章 Agent 通信协议](./chapter_protocol/README.md)
  - [17.1 MCP（Model Context Protocol）详解](./chapter_protocol/01_mcp_protocol.md)
  - [17.2 A2A（Agent-to-Agent）协议](./chapter_protocol/02_a2a_protocol.md)
  - [17.3 ANP（Agent Network Protocol）协议](./chapter_protocol/03_anp_protocol.md)
  - [17.4 Agent 间的消息传递与状态共享](./chapter_protocol/04_message_passing.md)
  - [17.5 实战：基于 MCP 的工具集成](./chapter_protocol/05_practice_mcp_integration.md)

---

- [第五部分：生产化篇](./part5.md)

- [第18章 Agent 的评估与优化](./chapter_evaluation/README.md)
  - [18.1 如何评估 Agent 的表现？](./chapter_evaluation/01_evaluation_methods.md)
  - [18.2 基准测试与评估指标](./chapter_evaluation/02_benchmarks.md)
  - [18.3 Prompt 调优策略](./chapter_evaluation/03_prompt_tuning.md)
  - [18.4 成本控制与性能优化](./chapter_evaluation/04_cost_optimization.md)
  - [18.5 可观测性：日志、追踪与监控](./chapter_evaluation/05_observability.md)

- [第19章 安全与可靠性](./chapter_security/README.md)
  - [19.1 Prompt 注入攻击与防御](./chapter_security/01_prompt_injection.md)
  - [19.2 幻觉问题与事实性保障](./chapter_security/02_hallucination.md)
  - [19.3 权限控制与沙箱隔离](./chapter_security/03_permission_sandbox.md)
  - [19.4 敏感数据保护](./chapter_security/04_data_protection.md)
  - [19.5 Agent 行为的可控性与对齐](./chapter_security/05_alignment.md)
  - [19.6 论文解读：安全与可靠性前沿研究](./chapter_security/06_paper_readings.md)

- [第20章 部署与生产化](./chapter_deployment/README.md)
  - [20.1 Agent 应用的部署架构](./chapter_deployment/01_deployment_architecture.md)
  - [20.2 API 服务化：FastAPI / Flask 封装](./chapter_deployment/02_api_service.md)
  - [20.3 容器化与云部署](./chapter_deployment/03_containerization.md)
  - [20.4 流式响应与并发处理](./chapter_deployment/04_streaming_concurrency.md)
  - [20.5 实战：部署一个生产级 Agent 服务](./chapter_deployment/05_practice_production_agent.md)

---

- [第六部分：综合项目篇](./part6.md)

- [第21章 项目实战：AI 编程助手](./chapter_coding_agent/README.md)
  - [21.1 项目架构设计](./chapter_coding_agent/01_architecture.md)
  - [21.2 代码理解与分析能力](./chapter_coding_agent/02_code_understanding.md)
  - [21.3 代码生成与修改能力](./chapter_coding_agent/03_code_generation.md)
  - [21.4 测试生成与 Bug 修复](./chapter_coding_agent/04_testing_debugging.md)
  - [21.5 完整项目实现](./chapter_coding_agent/05_full_implementation.md)

- [第22章 项目实战：智能数据分析 Agent](./chapter_data_agent/README.md)
  - [22.1 需求分析与架构设计](./chapter_data_agent/01_requirements.md)
  - [22.2 数据连接与查询](./chapter_data_agent/02_data_connection.md)
  - [22.3 自动化分析与可视化](./chapter_data_agent/03_analysis_visualization.md)
  - [22.4 报告生成与导出](./chapter_data_agent/04_report_generation.md)
  - [22.5 完整项目实现](./chapter_data_agent/05_full_implementation.md)

- [第23章 项目实战：多模态 Agent](./chapter_multimodal/README.md)
  - [23.1 多模态能力概述](./chapter_multimodal/01_multimodal_overview.md)
  - [23.2 图像理解与生成](./chapter_multimodal/02_image_understanding.md)
  - [23.3 语音交互集成](./chapter_multimodal/03_voice_interaction.md)
  - [23.4 实战：多模态个人助理](./chapter_multimodal/04_practice_multimodal_assistant.md)

---

- [附录]()

- [附录 A：常用 Prompt 模板大全](./appendix/prompt_templates.md)
- [附录 B：Agent 开发常见问题 FAQ](./appendix/faq.md)
- [附录 C：推荐学习资源与社区](./appendix/resources.md)
- [附录 D：术语表](./appendix/glossary.md)
- [附录 E：KL 散度详解](./appendix/kl_divergence.md)
