// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="preface.html">前言</a></span></li><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/index.html">第1章 什么是 Agent？</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/01_evolution.html">1.1 从聊天机器人到智能体的演进</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/02_core_concepts.html">1.2 Agent 的核心概念与定义</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/03_architecture.html">1.3 Agent 架构：感知-思考-行动循环</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/04_agent_vs_traditional.html">1.4 Agent 与传统程序的区别</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/05_use_cases.html">1.5 Agent 的应用场景全景图</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_intro/06_history.html">1.6 智能体发展史：从符号主义到大模型驱动</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/index.html">第2章 大语言模型基础</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/01_how_llm_works.html">2.1 LLM 是如何工作的？（直觉理解）</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/02_prompt_engineering.html">2.2 Prompt Engineering</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/03_prompting_strategies.html">2.3 Few-shot / Zero-shot / Chain-of-Thought 提示策略</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/04_api_basics.html">2.4 模型 API 调用入门</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_llm/05_model_parameters.html">2.5 Token、Temperature 与模型参数详解</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_context_engineering/index.html">第3章 上下文工程</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_context_engineering/01_context_vs_prompt.html">3.1 从提示工程到上下文工程</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_context_engineering/02_context_window.html">3.2 上下文窗口管理与注意力预算</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_context_engineering/03_long_horizon.html">3.3 长时程任务的上下文策略</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_context_engineering/04_practice_context_builder.html">3.4 实战：构建上下文管理器</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_setup/index.html">第4章 开发环境搭建</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_setup/01_python_setup.html">4.1 Python 环境与依赖管理</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_setup/02_install_libs.html">4.2 关键库安装</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_setup/03_api_key_management.html">4.3 API Key 管理与安全最佳实践</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_setup/04_hello_agent.html">4.4 第一个 Agent：Hello Agent！</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/index.html">第5章 工具调用（Tool Use / Function Calling）</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/01_why_tools.html">5.1 为什么 Agent 需要工具？</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/02_function_calling.html">5.2 Function Calling 机制</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/03_custom_tools.html">5.3 自定义工具的设计与实现</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/04_tool_description.html">5.4 工具描述的编写技巧</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/05_practice_search_calc.html">5.5 实战：搜索引擎 + 计算器 Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_tools/06_paper_readings.html">5.6 论文解读：工具学习前沿进展</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/index.html">第6章 Skill System</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/01_skill_overview.html">6.1 技能系统概述</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/02_skill_definition.html">6.2 技能的定义与封装</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/03_skill_learning.html">6.3 技能学习与获取</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/04_skill_discovery.html">6.4 技能发现与注册</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/05_practice_skill_system.html">6.5 实战：构建可复用的技能系统</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_skill/06_paper_readings.html">6.6 论文解读：技能系统前沿研究</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/index.html">第7章 记忆系统（Memory）</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/01_why_memory.html">7.1 为什么 Agent 需要记忆？</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/02_short_term_memory.html">7.2 短期记忆：对话历史管理</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/03_long_term_memory.html">7.3 长期记忆：向量数据库与检索</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/04_working_memory.html">7.4 工作记忆：Scratchpad 模式</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/05_practice_memory_agent.html">7.5 实战：带记忆的个人助理 Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_memory/06_paper_readings.html">7.6 论文解读：记忆系统前沿进展</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/index.html">第8章 规划与推理（Planning &amp; Reasoning）</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/01_how_agents_think.html">8.1 Agent 如何&quot;思考&quot;？</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/02_react_framework.html">8.2 ReAct：推理 + 行动框架</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/03_task_decomposition.html">8.3 任务分解：将复杂问题拆解为子任务</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/04_reflection.html">8.4 反思与自我纠错机制</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/05_practice_research_agent.html">8.5 实战：自动化研究助手 Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_planning/06_paper_readings.html">8.6 论文解读：规划与推理前沿研究</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/index.html">第9章 检索增强生成（RAG）</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/01_rag_concepts.html">9.1 RAG 的概念与工作原理</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/02_document_loading.html">9.2 文档加载与文本分割</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/03_embeddings_vectordb.html">9.3 向量嵌入与向量数据库</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/04_retrieval_strategies.html">9.4 检索策略与重排序</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/05_practice_qa_agent.html">9.5 实战：智能文档问答 Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_rag/06_paper_readings.html">9.6 论文解读：RAG 前沿进展</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/index.html">第10章 LangChain 深入实战</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/01_langchain_overview.html">10.1 LangChain 架构全景</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/02_chains.html">10.2 Chain：构建处理管道</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/03_langchain_agents.html">10.3 使用 LangChain 构建 Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/04_lcel.html">10.4 LCEL：LangChain 表达式语言</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langchain/05_practice_customer_service.html">10.5 实战：多功能客服 Agent</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/index.html">第11章 LangGraph：构建有状态的 Agent</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/01_why_graph.html">11.1 为什么需要图结构？</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/02_core_concepts.html">11.2 LangGraph 核心概念：节点、边、状态</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/03_first_graph_agent.html">11.3 构建你的第一个 Graph Agent</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/04_conditional_routing.html">11.4 条件路由与循环控制</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/05_human_in_the_loop.html">11.5 Human-in-the-Loop：人机协作</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_langgraph/06_practice_workflow_agent.html">11.6 实战：工作流自动化 Agent</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/index.html">第12章 其他主流框架概览</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/01_autogpt_babyagi.html">12.1 AutoGPT 与 BabyAGI 的启示</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/02_crewai.html">12.2 CrewAI：角色扮演型多 Agent 框架</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/03_autogen.html">12.3 AutoGen：多 Agent 对话框架</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/04_low_code_platforms.html">12.4 Dify / Coze 等低代码 Agent 平台</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_frameworks/05_how_to_choose.html">12.5 如何选择合适的框架？</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/index.html">第13章 多 Agent 协作</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/01_single_agent_limits.html">13.1 单 Agent 的局限性</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/02_communication_patterns.html">13.2 多 Agent 通信模式</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/03_role_assignment.html">13.3 角色分工与任务分配</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/04_supervisor_vs_decentralized.html">13.4 Supervisor 模式 vs. 去中心化模式</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/05_practice_dev_team.html">13.5 实战：多 Agent 软件开发团队</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multi_agent/06_paper_readings.html">13.6 论文解读：多 Agent 系统前沿研究</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/index.html">第14章 Agent 通信协议</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/01_mcp_protocol.html">14.1 MCP（Model Context Protocol）详解</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/02_a2a_protocol.html">14.2 A2A（Agent-to-Agent）协议</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/03_anp_protocol.html">14.3 ANP（Agent Network Protocol）协议</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/04_message_passing.html">14.4 Agent 间的消息传递与状态共享</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_protocol/05_practice_mcp_integration.html">14.5 实战：基于 MCP 的工具集成</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/index.html">第15章 Agent 的评估与优化</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/01_evaluation_methods.html">15.1 如何评估 Agent 的表现？</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/02_benchmarks.html">15.2 基准测试与评估指标</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/03_prompt_tuning.html">15.3 Prompt 调优策略</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/04_cost_optimization.html">15.4 成本控制与性能优化</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_evaluation/05_observability.html">15.5 可观测性：日志、追踪与监控</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/index.html">第16章 安全与可靠性</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/01_prompt_injection.html">16.1 Prompt 注入攻击与防御</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/02_hallucination.html">16.2 幻觉问题与事实性保障</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/03_permission_sandbox.html">16.3 权限控制与沙箱隔离</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/04_data_protection.html">16.4 敏感数据保护</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/05_alignment.html">16.5 Agent 行为的可控性与对齐</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_security/06_paper_readings.html">16.6 论文解读：安全与可靠性前沿研究</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/index.html">第17章 部署与生产化</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/01_deployment_architecture.html">17.1 Agent 应用的部署架构</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/02_api_service.html">17.2 API 服务化：FastAPI / Flask 封装</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/03_containerization.html">17.3 容器化与云部署</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/04_streaming_concurrency.html">17.4 流式响应与并发处理</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_deployment/05_practice_production_agent.html">17.5 实战：部署一个生产级 Agent 服务</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/index.html">第18章 Agentic-RL：智能体强化学习训练</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/01_agentic_rl_overview.html">18.1 什么是 Agentic-RL</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/02_sft_lora.html">18.2 SFT + LoRA 基础训练</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/03_grpo.html">18.3 策略优化算法详解：PPO、DPO 与 GRPO</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/04_reward_design.html">18.4 奖励函数设计</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_agentic_rl/05_practice_training.html">18.5 实战：完整 SFT + GRPO 训练 Pipeline</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/index.html">第19章 项目实战：AI 编程助手</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/01_architecture.html">19.1 项目架构设计</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/02_code_understanding.html">19.2 代码理解与分析能力</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/03_code_generation.html">19.3 代码生成与修改能力</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/04_testing_debugging.html">19.4 测试生成与 Bug 修复</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_coding_agent/05_full_implementation.html">19.5 完整项目实现</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/index.html">第20章 项目实战：智能数据分析 Agent</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/01_requirements.html">20.1 需求分析与架构设计</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/02_data_connection.html">20.2 数据连接与查询</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/03_analysis_visualization.html">20.3 自动化分析与可视化</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/04_report_generation.html">20.4 报告生成与导出</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_data_agent/05_full_implementation.html">20.5 完整项目实现</a></span></li></ol><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multimodal/index.html">第21章 项目实战：多模态 Agent</a></span><ol class="section"><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multimodal/01_multimodal_overview.html">21.1 多模态能力概述</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multimodal/02_image_understanding.html">21.2 图像理解与生成</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multimodal/03_voice_interaction.html">21.3 语音交互集成</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="chapter_multimodal/04_practice_multimodal_assistant.html">21.4 实战：多模态个人助理</a></span></li></ol><li class="chapter-item expanded "><li class="spacer"></li></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="appendix/prompt_templates.html">附录 A：常用 Prompt 模板大全</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="appendix/faq.html">附录 B：Agent 开发常见问题 FAQ</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="appendix/resources.html">附录 C：推荐学习资源与社区</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="appendix/glossary.html">附录 D：术语表</a></span></li><li class="chapter-item expanded "><span class="chapter-link-wrapper"><a href="appendix/kl_divergence.html">附录 E：KL 散度详解</a></span></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString().split('#')[0].split('?')[0];
        if (current_page.endsWith('/')) {
            current_page += 'index.html';
        }
        const links = Array.prototype.slice.call(this.querySelectorAll('a'));
        const l = links.length;
        for (let i = 0; i < l; ++i) {
            const link = links[i];
            const href = link.getAttribute('href');
            if (href && !href.startsWith('#') && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The 'index' page is supposed to alias the first chapter in the book.
            if (link.href === current_page
                || i === 0
                && path_to_root === ''
                && current_page.endsWith('/index.html')) {
                link.classList.add('active');
                let parent = link.parentElement;
                while (parent) {
                    if (parent.tagName === 'LI' && parent.classList.contains('chapter-item')) {
                        parent.classList.add('expanded');
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', e => {
            if (e.target.tagName === 'A') {
                const clientRect = e.target.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                sessionStorage.setItem('sidebar-scroll-offset', clientRect.top - sidebarRect.top);
            }
        }, { passive: true });
        const sidebarScrollOffset = sessionStorage.getItem('sidebar-scroll-offset');
        sessionStorage.removeItem('sidebar-scroll-offset');
        if (sidebarScrollOffset !== null) {
            // preserve sidebar scroll position when navigating via links within sidebar
            const activeSection = this.querySelector('.active');
            if (activeSection) {
                const clientRect = activeSection.getBoundingClientRect();
                const sidebarRect = this.getBoundingClientRect();
                const currentOffset = clientRect.top - sidebarRect.top;
                this.scrollTop += currentOffset - parseFloat(sidebarScrollOffset);
            }
        } else {
            // scroll sidebar to current active section when navigating via
            // 'next/previous chapter' buttons
            const activeSection = document.querySelector('#mdbook-sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        const sidebarAnchorToggles = document.querySelectorAll('.chapter-fold-toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(el => {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define('mdbook-sidebar-scrollbox', MDBookSidebarScrollbox);


// ---------------------------------------------------------------------------
// Support for dynamically adding headers to the sidebar.

(function() {
    // This is used to detect which direction the page has scrolled since the
    // last scroll event.
    let lastKnownScrollPosition = 0;
    // This is the threshold in px from the top of the screen where it will
    // consider a header the "current" header when scrolling down.
    const defaultDownThreshold = 150;
    // Same as defaultDownThreshold, except when scrolling up.
    const defaultUpThreshold = 300;
    // The threshold is a virtual horizontal line on the screen where it
    // considers the "current" header to be above the line. The threshold is
    // modified dynamically to handle headers that are near the bottom of the
    // screen, and to slightly offset the behavior when scrolling up vs down.
    let threshold = defaultDownThreshold;
    // This is used to disable updates while scrolling. This is needed when
    // clicking the header in the sidebar, which triggers a scroll event. It
    // is somewhat finicky to detect when the scroll has finished, so this
    // uses a relatively dumb system of disabling scroll updates for a short
    // time after the click.
    let disableScroll = false;
    // Array of header elements on the page.
    let headers;
    // Array of li elements that are initially collapsed headers in the sidebar.
    // I'm not sure why eslint seems to have a false positive here.
    // eslint-disable-next-line prefer-const
    let headerToggles = [];
    // This is a debugging tool for the threshold which you can enable in the console.
    let thresholdDebug = false;

    // Updates the threshold based on the scroll position.
    function updateThreshold() {
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;

        // The number of pixels below the viewport, at most documentHeight.
        // This is used to push the threshold down to the bottom of the page
        // as the user scrolls towards the bottom.
        const pixelsBelow = Math.max(0, documentHeight - (scrollTop + windowHeight));
        // The number of pixels above the viewport, at least defaultDownThreshold.
        // Similar to pixelsBelow, this is used to push the threshold back towards
        // the top when reaching the top of the page.
        const pixelsAbove = Math.max(0, defaultDownThreshold - scrollTop);
        // How much the threshold should be offset once it gets close to the
        // bottom of the page.
        const bottomAdd = Math.max(0, windowHeight - pixelsBelow - defaultDownThreshold);
        let adjustedBottomAdd = bottomAdd;

        // Adjusts bottomAdd for a small document. The calculation above
        // assumes the document is at least twice the windowheight in size. If
        // it is less than that, then bottomAdd needs to be shrunk
        // proportional to the difference in size.
        if (documentHeight < windowHeight * 2) {
            const maxPixelsBelow = documentHeight - windowHeight;
            const t = 1 - pixelsBelow / Math.max(1, maxPixelsBelow);
            const clamp = Math.max(0, Math.min(1, t));
            adjustedBottomAdd *= clamp;
        }

        let scrollingDown = true;
        if (scrollTop < lastKnownScrollPosition) {
            scrollingDown = false;
        }

        if (scrollingDown) {
            // When scrolling down, move the threshold up towards the default
            // downwards threshold position. If near the bottom of the page,
            // adjustedBottomAdd will offset the threshold towards the bottom
            // of the page.
            const amountScrolledDown = scrollTop - lastKnownScrollPosition;
            const adjustedDefault = defaultDownThreshold + adjustedBottomAdd;
            threshold = Math.max(adjustedDefault, threshold - amountScrolledDown);
        } else {
            // When scrolling up, move the threshold down towards the default
            // upwards threshold position. If near the bottom of the page,
            // quickly transition the threshold back up where it normally
            // belongs.
            const amountScrolledUp = lastKnownScrollPosition - scrollTop;
            const adjustedDefault = defaultUpThreshold - pixelsAbove
                + Math.max(0, adjustedBottomAdd - defaultDownThreshold);
            threshold = Math.min(adjustedDefault, threshold + amountScrolledUp);
        }

        if (documentHeight <= windowHeight) {
            threshold = 0;
        }

        if (thresholdDebug) {
            const id = 'mdbook-threshold-debug-data';
            let data = document.getElementById(id);
            if (data === null) {
                data = document.createElement('div');
                data.id = id;
                data.style.cssText = `
                    position: fixed;
                    top: 50px;
                    right: 10px;
                    background-color: 0xeeeeee;
                    z-index: 9999;
                    pointer-events: none;
                `;
                document.body.appendChild(data);
            }
            data.innerHTML = `
                <table>
                  <tr><td>documentHeight</td><td>${documentHeight.toFixed(1)}</td></tr>
                  <tr><td>windowHeight</td><td>${windowHeight.toFixed(1)}</td></tr>
                  <tr><td>scrollTop</td><td>${scrollTop.toFixed(1)}</td></tr>
                  <tr><td>pixelsAbove</td><td>${pixelsAbove.toFixed(1)}</td></tr>
                  <tr><td>pixelsBelow</td><td>${pixelsBelow.toFixed(1)}</td></tr>
                  <tr><td>bottomAdd</td><td>${bottomAdd.toFixed(1)}</td></tr>
                  <tr><td>adjustedBottomAdd</td><td>${adjustedBottomAdd.toFixed(1)}</td></tr>
                  <tr><td>scrollingDown</td><td>${scrollingDown}</td></tr>
                  <tr><td>threshold</td><td>${threshold.toFixed(1)}</td></tr>
                </table>
            `;
            drawDebugLine();
        }

        lastKnownScrollPosition = scrollTop;
    }

    function drawDebugLine() {
        if (!document.body) {
            return;
        }
        const id = 'mdbook-threshold-debug-line';
        const existingLine = document.getElementById(id);
        if (existingLine) {
            existingLine.remove();
        }
        const line = document.createElement('div');
        line.id = id;
        line.style.cssText = `
            position: fixed;
            top: ${threshold}px;
            left: 0;
            width: 100vw;
            height: 2px;
            background-color: red;
            z-index: 9999;
            pointer-events: none;
        `;
        document.body.appendChild(line);
    }

    function mdbookEnableThresholdDebug() {
        thresholdDebug = true;
        updateThreshold();
        drawDebugLine();
    }

    window.mdbookEnableThresholdDebug = mdbookEnableThresholdDebug;

    // Updates which headers in the sidebar should be expanded. If the current
    // header is inside a collapsed group, then it, and all its parents should
    // be expanded.
    function updateHeaderExpanded(currentA) {
        // Add expanded to all header-item li ancestors.
        let current = currentA.parentElement;
        while (current) {
            if (current.tagName === 'LI' && current.classList.contains('header-item')) {
                current.classList.add('expanded');
            }
            current = current.parentElement;
        }
    }

    // Updates which header is marked as the "current" header in the sidebar.
    // This is done with a virtual Y threshold, where headers at or below
    // that line will be considered the current one.
    function updateCurrentHeader() {
        if (!headers || !headers.length) {
            return;
        }

        // Reset the classes, which will be rebuilt below.
        const els = document.getElementsByClassName('current-header');
        for (const el of els) {
            el.classList.remove('current-header');
        }
        for (const toggle of headerToggles) {
            toggle.classList.remove('expanded');
        }

        // Find the last header that is above the threshold.
        let lastHeader = null;
        for (const header of headers) {
            const rect = header.getBoundingClientRect();
            if (rect.top <= threshold) {
                lastHeader = header;
            } else {
                break;
            }
        }
        if (lastHeader === null) {
            lastHeader = headers[0];
            const rect = lastHeader.getBoundingClientRect();
            const windowHeight = window.innerHeight;
            if (rect.top >= windowHeight) {
                return;
            }
        }

        // Get the anchor in the summary.
        const href = '#' + lastHeader.id;
        const a = [...document.querySelectorAll('.header-in-summary')]
            .find(element => element.getAttribute('href') === href);
        if (!a) {
            return;
        }

        a.classList.add('current-header');

        updateHeaderExpanded(a);
    }

    // Updates which header is "current" based on the threshold line.
    function reloadCurrentHeader() {
        if (disableScroll) {
            return;
        }
        updateThreshold();
        updateCurrentHeader();
    }


    // When clicking on a header in the sidebar, this adjusts the threshold so
    // that it is located next to the header. This is so that header becomes
    // "current".
    function headerThresholdClick(event) {
        // See disableScroll description why this is done.
        disableScroll = true;
        setTimeout(() => {
            disableScroll = false;
        }, 100);
        // requestAnimationFrame is used to delay the update of the "current"
        // header until after the scroll is done, and the header is in the new
        // position.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                // Closest is needed because if it has child elements like <code>.
                const a = event.target.closest('a');
                const href = a.getAttribute('href');
                const targetId = href.substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    threshold = targetElement.getBoundingClientRect().bottom;
                    updateCurrentHeader();
                }
            });
        });
    }

    // Takes the nodes from the given head and copies them over to the
    // destination, along with some filtering.
    function filterHeader(source, dest) {
        const clone = source.cloneNode(true);
        clone.querySelectorAll('mark').forEach(mark => {
            mark.replaceWith(...mark.childNodes);
        });
        dest.append(...clone.childNodes);
    }

    // Scans page for headers and adds them to the sidebar.
    document.addEventListener('DOMContentLoaded', function() {
        const activeSection = document.querySelector('#mdbook-sidebar .active');
        if (activeSection === null) {
            return;
        }

        const main = document.getElementsByTagName('main')[0];
        headers = Array.from(main.querySelectorAll('h2, h3, h4, h5, h6'))
            .filter(h => h.id !== '' && h.children.length && h.children[0].tagName === 'A');

        if (headers.length === 0) {
            return;
        }

        // Build a tree of headers in the sidebar.

        const stack = [];

        const firstLevel = parseInt(headers[0].tagName.charAt(1));
        for (let i = 1; i < firstLevel; i++) {
            const ol = document.createElement('ol');
            ol.classList.add('section');
            if (stack.length > 0) {
                stack[stack.length - 1].ol.appendChild(ol);
            }
            stack.push({level: i + 1, ol: ol});
        }

        // The level where it will start folding deeply nested headers.
        const foldLevel = 3;

        for (let i = 0; i < headers.length; i++) {
            const header = headers[i];
            const level = parseInt(header.tagName.charAt(1));

            const currentLevel = stack[stack.length - 1].level;
            if (level > currentLevel) {
                // Begin nesting to this level.
                for (let nextLevel = currentLevel + 1; nextLevel <= level; nextLevel++) {
                    const ol = document.createElement('ol');
                    ol.classList.add('section');
                    const last = stack[stack.length - 1];
                    const lastChild = last.ol.lastChild;
                    // Handle the case where jumping more than one nesting
                    // level, which doesn't have a list item to place this new
                    // list inside of.
                    if (lastChild) {
                        lastChild.appendChild(ol);
                    } else {
                        last.ol.appendChild(ol);
                    }
                    stack.push({level: nextLevel, ol: ol});
                }
            } else if (level < currentLevel) {
                while (stack.length > 1 && stack[stack.length - 1].level > level) {
                    stack.pop();
                }
            }

            const li = document.createElement('li');
            li.classList.add('header-item');
            li.classList.add('expanded');
            if (level < foldLevel) {
                li.classList.add('expanded');
            }
            const span = document.createElement('span');
            span.classList.add('chapter-link-wrapper');
            const a = document.createElement('a');
            span.appendChild(a);
            a.href = '#' + header.id;
            a.classList.add('header-in-summary');
            filterHeader(header.children[0], a);
            a.addEventListener('click', headerThresholdClick);
            const nextHeader = headers[i + 1];
            if (nextHeader !== undefined) {
                const nextLevel = parseInt(nextHeader.tagName.charAt(1));
                if (nextLevel > level && level >= foldLevel) {
                    const toggle = document.createElement('a');
                    toggle.classList.add('chapter-fold-toggle');
                    toggle.classList.add('header-toggle');
                    toggle.addEventListener('click', () => {
                        li.classList.toggle('expanded');
                    });
                    const toggleDiv = document.createElement('div');
                    toggleDiv.textContent = '❱';
                    toggle.appendChild(toggleDiv);
                    span.appendChild(toggle);
                    headerToggles.push(li);
                }
            }
            li.appendChild(span);

            const currentParent = stack[stack.length - 1];
            currentParent.ol.appendChild(li);
        }

        const onThisPage = document.createElement('div');
        onThisPage.classList.add('on-this-page');
        onThisPage.append(stack[0].ol);
        const activeItemSpan = activeSection.parentElement;
        activeItemSpan.after(onThisPage);
    });

    document.addEventListener('DOMContentLoaded', reloadCurrentHeader);
    document.addEventListener('scroll', reloadCurrentHeader, { passive: true });
})();

