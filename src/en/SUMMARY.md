# Learn Agent Development from Scratch

[Preface](./preface.md)

---

- [Part I: Getting Started](./part1.md)

- [Chapter 1: What Is an Agent?](./chapter_intro/README.md)
  - [1.1 From Chatbots to Intelligent Agents](./chapter_intro/01_evolution.md)
  - [1.2 Core Concepts and Definitions of Agents](./chapter_intro/02_core_concepts.md)
  - [1.3 Agent Architecture: The Perception-Thinking-Action Loop](./chapter_intro/03_architecture.md)
  - [1.4 Agents vs. Traditional Programs](./chapter_intro/04_agent_vs_traditional.md)
  - [1.5 Landscape of Agent Applications](./chapter_intro/05_use_cases.md)
  - [1.6 History of Agents: From Symbolic AI to LLM-Driven](./chapter_intro/06_history.md)

- [Chapter 2: Large Language Model Fundamentals](./chapter_llm/README.md)
  - [2.1 How LLMs Work (Intuitive Understanding)](./chapter_llm/01_how_llm_works.md)
  - [2.2 Prompt Engineering](./chapter_llm/02_prompt_engineering.md)
  - [2.3 Few-shot / Zero-shot / Chain-of-Thought Prompting Strategies](./chapter_llm/03_prompting_strategies.md)
  - [2.4 Introduction to Model API Calls](./chapter_llm/04_api_basics.md)
  - [2.5 Tokens, Temperature, and Model Parameters](./chapter_llm/05_model_parameters.md)
  - [2.6 Frontier Foundation Models and Selection Guide](./chapter_llm/06_foundation_model_landscape.md)
  - [2.7 Foundation Model Architecture Deep Dive](./chapter_llm/07_model_architecture.md)
  - [2.8 SFT and Reinforcement Learning Training Data Preparation](./chapter_llm/08_training_data.md)

---

- [Part II: Core Capabilities](./part2.md)

- [Chapter 3: Tool Use / Function Calling](./chapter_tools/README.md)
  - [3.1 Why Do Agents Need Tools?](./chapter_tools/01_why_tools.md)
  - [3.2 Function Calling Mechanism](./chapter_tools/02_function_calling.md)
  - [3.3 Designing and Implementing Custom Tools](./chapter_tools/03_custom_tools.md)
  - [3.4 Writing Effective Tool Descriptions](./chapter_tools/04_tool_description.md)
  - [3.5 Practice: Search Engine + Calculator Agent](./chapter_tools/05_practice_search_calc.md)
  - [3.6 Paper Reading: Frontiers in Tool Learning](./chapter_tools/06_paper_readings.md)

- [Chapter 4: Memory Systems](./chapter_memory/README.md)
  - [4.1 Why Do Agents Need Memory?](./chapter_memory/01_why_memory.md)
  - [4.2 Short-Term Memory: Conversation History Management](./chapter_memory/02_short_term_memory.md)
  - [4.3 Long-Term Memory: Vector Databases and Retrieval](./chapter_memory/03_long_term_memory.md)
  - [4.4 Working Memory: Scratchpad Pattern](./chapter_memory/04_working_memory.md)
  - [4.5 Practice: Personal Assistant Agent with Memory](./chapter_memory/05_practice_memory_agent.md)
  - [4.6 Paper Reading: Frontiers in Memory Systems](./chapter_memory/06_paper_readings.md)
  - [4.7 Practice: MemGPT/Letta Memory Architecture](./chapter_memory/06b_memgpt_practice.md)

- [Chapter 5: Planning and Reasoning](./chapter_planning/README.md)
  - [5.1 How Do Agents "Think"?](./chapter_planning/01_how_agents_think.md)
  - [5.2 ReAct: Reasoning + Acting Framework](./chapter_planning/02_react_framework.md)
  - [5.3 Task Decomposition: Breaking Complex Problems into Subtasks](./chapter_planning/03_task_decomposition.md)
  - [5.4 Reflection and Self-Correction Mechanisms](./chapter_planning/04_reflection.md)
  - [5.5 Practice: Automated Research Assistant Agent](./chapter_planning/05_practice_research_agent.md)
  - [5.6 Plan-and-Execute and Test-time Compute Scaling](./chapter_planning/07_plan_and_execute.md)
  - [5.7 Paper Reading: Frontiers in Planning and Reasoning](./chapter_planning/06_paper_readings.md)

- [Chapter 6: Retrieval-Augmented Generation (RAG)](./chapter_rag/README.md)
  - [6.1 RAG Concepts and How It Works](./chapter_rag/01_rag_concepts.md)
  - [6.2 Document Loading and Text Splitting](./chapter_rag/02_document_loading.md)
  - [6.3 Vector Embeddings and Vector Databases](./chapter_rag/03_embeddings_vectordb.md)
  - [6.4 Retrieval Strategies and Reranking](./chapter_rag/04_retrieval_strategies.md)
  - [6.5 Practice: Intelligent Document Q&A Agent](./chapter_rag/05_practice_qa_agent.md)
  - [6.6 Paper Reading: Frontiers in RAG](./chapter_rag/06_paper_readings.md)
  - [6.7 Advanced RAG: GraphRAG and Agentic RAG Engineering Practice](./chapter_rag/07_advanced_rag.md)

- [Chapter 7: Context Engineering](./chapter_context_engineering/README.md)
  - [7.1 From Prompt Engineering to Context Engineering](./chapter_context_engineering/01_context_vs_prompt.md)
  - [7.2 Context Window Management and Attention Budget](./chapter_context_engineering/02_context_window.md)
  - [7.3 Context Strategies for Long-Horizon Tasks](./chapter_context_engineering/03_long_horizon.md)
  - [7.4 Practice: Building a Context Manager](./chapter_context_engineering/04_practice_context_builder.md)
  - [7.5 Latest Advances in Context Engineering](./chapter_context_engineering/05_latest_advances.md)

- [Chapter 8: Harness Engineering: System Engineering for Controlling Agents](./chapter_harness/README.md)
  - [8.1 What Is Harness Engineering?](./chapter_harness/01_what_is_harness.md)
  - [8.2 The Six Engineering Pillars](./chapter_harness/02_six_pillars.md)
  - [8.3 AGENTS.md / CLAUDE.md: Writing Your Agent Constitution](./chapter_harness/03_agents_md.md)
  - [8.4 Production Case Studies: OpenAI, LangChain, Stripe](./chapter_harness/04_production_cases.md)
  - [8.5 Practice: Building Your First Harness System](./chapter_harness/05_practice_harness_builder.md)
  - [8.6 Structured Output: Engineering Reliable JSON](./chapter_harness/06_structured_output.md)

- [Chapter 9: Skill System](./chapter_skill/README.md)
  - [9.1 Skill System Overview](./chapter_skill/01_skill_overview.md)
  - [9.2 Skill Definition and Encapsulation](./chapter_skill/02_skill_definition.md)
  - [9.3 Skill Learning and Acquisition](./chapter_skill/03_skill_learning.md)
  - [9.4 Skill Discovery and Registration](./chapter_skill/04_skill_discovery.md)
  - [9.5 Practice: Building a Reusable Skill System](./chapter_skill/05_practice_skill_system.md)
  - [9.6 Paper Reading: Frontiers in Skill Systems](./chapter_skill/06_paper_readings.md)
  - [9.7 Tool, Skill & Sub-Agent: Three-Layer Capability Abstraction](./chapter_skill/07_tool_skill_subagent.md)
  - [9.8 Skills Bible: Superpowers Engineering Practice Guide](./chapter_skill/08_superpowers_guide.md)

- [Chapter 10: Agentic-RL: Reinforcement Learning for Agents](./chapter_agentic_rl/README.md)
  - [10.1 What Is Agentic-RL?](./chapter_agentic_rl/01_agentic_rl_overview.md)
  - [10.2 SFT + LoRA Fundamentals](./chapter_agentic_rl/02_sft_lora.md)
  - [10.2b Distributed Training Basics: DP / TP / PP / SP / ZeRO](./chapter_agentic_rl/02b_distributed_training.md)
  - [10.3 PPO: Proximal Policy Optimization](./chapter_agentic_rl/03_ppo.md)
  - [10.4 DPO: Direct Preference Optimization](./chapter_agentic_rl/04_dpo.md)
  - [10.5 GRPO/GSPO: Group Relative Policy Optimization and Reward Design](./chapter_agentic_rl/05_grpo.md)
  - [10.6 Practice: Complete SFT + GRPO Training Pipeline](./chapter_agentic_rl/06_practice_training.md)
  - [10.7 Latest Research Progress (2025–2026)](./chapter_agentic_rl/07_latest_research.md)
  - [10.8 Agent-Specific Fine-Tuning: Teaching Models to Use Tools](./chapter_agentic_rl/08_agent_finetuning.md)

---

- [Part III: Framework Practice](./part3.md)

- [Chapter 12: LangChain In-Depth](./chapter_langchain/README.md)
  - [12.1 LangChain Architecture Overview](./chapter_langchain/01_langchain_overview.md)
  - [12.2 Chains: Building Processing Pipelines](./chapter_langchain/02_chains.md)
  - [12.3 Building Agents with LangChain](./chapter_langchain/03_langchain_agents.md)
  - [12.4 LCEL: LangChain Expression Language](./chapter_langchain/04_lcel.md)
  - [12.5 Practice: Multi-Function Customer Service Agent](./chapter_langchain/05_practice_customer_service.md)
  - [12.6 LangSmith Integration and Observability](./chapter_langchain/06_langsmith_integration.md)
  - [12.7 LangChain Ecosystem 2026](./chapter_langchain/07_langchain_ecosystem_2026.md)
  - [12.8 LangChain Production Patterns](./chapter_langchain/08_production_patterns.md)

- [Chapter 13: LangGraph: Building Stateful Agents](./chapter_langgraph/README.md)
  - [13.1 Why Graph Structures?](./chapter_langgraph/01_why_graph.md)
  - [13.2 LangGraph Core Concepts: Nodes, Edges, and State](./chapter_langgraph/02_core_concepts.md)
  - [13.3 Build Your First Graph Agent](./chapter_langgraph/03_first_graph_agent.md)
  - [13.4 Conditional Routing and Loop Control](./chapter_langgraph/04_conditional_routing.md)
  - [13.5 Human-in-the-Loop: Human-AI Collaboration](./chapter_langgraph/05_human_in_the_loop.md)
  - [13.6 Practice: Workflow Automation Agent](./chapter_langgraph/06_practice_workflow_agent.md)

- [Chapter 14: OpenClaw — Cross-Platform Personal AI Assistant](./chapter_openclaw/README.md)
  - [14.1 OpenClaw Panorama: From Clawdbot to OpenClaw](./chapter_openclaw/01_history_and_positioning.md)
  - [14.2 Installation & 4 Deployment Modes](./chapter_openclaw/02_install_and_deploy.md)
  - [14.3 Architecture: Gateway / Agent Loop / Skills](./chapter_openclaw/03_architecture.md)
  - [14.4 Multi-Channel Routing: WhatsApp / Telegram / Discord / Slack / Signal](./chapter_openclaw/04_channels.md)
  - [14.5 Skills & Plugin Ecosystem: ClawHub and Community Contributions](./chapter_openclaw/05_skills_and_plugins.md)
  - [14.6 Practice: Build a Personal Assistant](./chapter_openclaw/06_practice.md)
  - [14.7 Lessons for Engineers](./chapter_openclaw/07_lessons_for_engineers.md)

- [Chapter 15: Hermes Agent — The Self-Evolving Agent](./chapter_hermes/README.md)
  - [15.1 Birth & the "Growing Agent" Philosophy](./chapter_hermes/01_birth_and_philosophy.md)
  - [15.2 Installation & Migration (from OpenClaw)](./chapter_hermes/02_install_and_migration.md)
  - [15.3 Three-Layer Architecture: Gateway / Engine / Plugin](./chapter_hermes/03_three_layer_architecture.md)
  - [15.4 Core: Self-Evolving Skills Loop](./chapter_hermes/04_self_evolving_skills.md)
  - [15.5 Three-Layer Memory](./chapter_hermes/05_memory.md)
  - [15.6 Nudge Engine & Cross-Session Learning](./chapter_hermes/06_nudge_engine.md)
  - [15.7 Borrowing the Philosophy](./chapter_hermes/07_lessons_philosophy.md)

- [Chapter 16: Deep Dive into Claude Code](./chapter_claude_code/README.md)
  - [16.1 The Prehistory of Industrial Harnesses: From AutoGPT to Claude Code](./chapter_claude_code/01_industry_history.md)
  - [16.2 Getting to Know Claude Code](./chapter_claude_code/02_introduction.md)
  - [16.3 Deep Dive into Core Architecture](./chapter_claude_code/03_architecture.md)
  - [16.4 System Prompt, Permission Engineering & Prompt Cache](./chapter_claude_code/04_system_prompt_and_permissions.md)
  - [16.5 Advanced Usage: MCP, Hooks, and Skills](./chapter_claude_code/05_advanced_usage.md)
  - [16.6 Production Practice](./chapter_claude_code/06_production_and_team.md)

- [Chapter 17: DeepSeek Harness — The Everything-Is-a-Plugin Foundation](./chapter_deepseek_harness/README.md)
  - [17.1 What Is DeepSeek Harness: Cordis Kernel and "Everything Is a Plugin"](./chapter_deepseek_harness/01_what_is_dsh.md)
  - [17.2 Installation & 4 Run Modes](./chapter_deepseek_harness/02_install_and_modes.md)
  - [17.3 Architecture: Cordis Microkernel & Plugin Topology](./chapter_deepseek_harness/03_cordis_architecture.md)
  - [17.4 Plugin Development: tool / llm / skill / subagent](./chapter_deepseek_harness/04_plugin_development.md)
  - [17.5 Comparison: DSH vs Claude Code / OpenClaw / Hermes](./chapter_deepseek_harness/05_comparison.md)
  - [17.6 Borrowing the Philosophy](./chapter_deepseek_harness/06_lessons_philosophy.md)
  - [17.7 Summary: The 6-Harness Decision Matrix](./chapter_deepseek_harness/07_summary_and_decision_matrix.md)

---

- [Part IV: Multi-Agent Systems](./part4.md)

- [Chapter 18: Multi-Agent Collaboration](./chapter_18_multi_agent/README.md)
  - [18.1 Limitations of Single Agents](./chapter_18_multi_agent/01_single_agent_limits.md)
  - [18.2 Multi-Agent Communication Patterns](./chapter_18_multi_agent/02_communication_patterns.md)
  - [18.3 Role Assignment and Task Allocation](./chapter_18_multi_agent/03_role_assignment.md)
  - [18.4 Supervisor Mode vs. Decentralized Mode](./chapter_18_multi_agent/04_supervisor_vs_decentralized.md)
  - [18.5 Practice: Multi-Agent Software Development Team](./chapter_18_multi_agent/05_practice_dev_team.md)
  - [18.6 Paper Reading: Frontiers in Multi-Agent Systems](./chapter_18_multi_agent/06_paper_readings.md)
  - [18.7 Framework Supplement: CrewAI (Role-Playing Multi-Agent Framework)](./chapter_18_multi_agent/07_crewai.md)

- [Chapter 19: Agent Communication Protocols](./chapter_19_protocol/README.md)
  - [19.1 MCP (Model Context Protocol) Explained](./chapter_19_protocol/01_mcp_protocol.md)
  - [19.2 A2A (Agent-to-Agent) Protocol](./chapter_19_protocol/02_a2a_protocol.md)
  - [19.3 ANP (Agent Network Protocol)](./chapter_19_protocol/03_anp_protocol.md)
  - [19.4 Message Passing and State Sharing Between Agents](./chapter_19_protocol/04_message_passing.md)
  - [19.5 Practice: Tool Integration Based on MCP](./chapter_19_protocol/05_practice_mcp_integration.md)
  - [19.6 Framework Supplement: AutoGen (Multi-Agent Dialogue Framework)](./chapter_19_protocol/06_autogen.md)

---

- [Part V: Production](./part5.md)

- [Chapter 20: Agent Evaluation and Optimization](./chapter_20_evaluation/README.md)
  - [20.1 How to Evaluate Agent Performance?](./chapter_20_evaluation/01_evaluation_methods.md)
  - [20.2 Benchmarks and Evaluation Metrics](./chapter_20_evaluation/02_benchmarks.md)
  - [20.3 Prompt Tuning Strategies](./chapter_20_evaluation/03_prompt_tuning.md)
  - [20.4 Cost Control and Performance Optimization](./chapter_20_evaluation/04_cost_optimization.md)
  - [20.5 Observability: Logging, Tracing, and Monitoring](./chapter_20_evaluation/05_observability.md)
  - [20.6 Agent-Specific Evaluation Frameworks](./chapter_20_evaluation/06_agent_evaluation.md)
  - [20.7 A/B Testing and Regression Test Automation](./chapter_20_evaluation/07_ab_testing.md)
  - [20.8 Model Routing Evaluation](./chapter_20_evaluation/08_model_routing.md)

- [Chapter 21: Security and Reliability](./chapter_21_security/README.md)
  - [21.1 Prompt Injection Attacks and Defenses](./chapter_21_security/01_prompt_injection.md)
  - [21.2 Hallucination Problems and Factuality Assurance](./chapter_21_security/02_hallucination.md)
  - [21.3 Permission Control and Sandbox Isolation](./chapter_21_security/03_permission_sandbox.md)
  - [21.4 Sensitive Data Protection](./chapter_21_security/04_data_protection.md)
  - [21.5 Controllability and Alignment of Agent Behavior](./chapter_21_security/05_alignment.md)
  - [21.6 Paper Reading: Frontiers in Security and Reliability](./chapter_21_security/06_paper_readings.md)
  - [21.7 Guardrails Runtime Protection](./chapter_21_security/07_guardrails_runtime.md)
  - [21.8 Red Teaming Methodology](./chapter_21_security/08_red_teaming.md)

- [Chapter 22: Deployment and Productionization](./chapter_22_deployment/README.md)
  - [22.1 Deployment Architecture for Agent Applications](./chapter_22_deployment/01_deployment_architecture.md)
  - [22.2 API Service: FastAPI / Flask Wrapping](./chapter_22_deployment/02_api_service.md)
  - [22.3 Containerization and Cloud Deployment](./chapter_22_deployment/03_containerization.md)
  - [22.4 Streaming Responses and Concurrent Processing](./chapter_22_deployment/04_streaming_concurrency.md)
  - [22.5 Practice: Deploying a Production-Grade Agent Service](./chapter_22_deployment/05_practice_production_agent.md)
  - [22.6 Model Inference Serving](./chapter_22_deployment/06_model_serving.md)
  - [22.7 Kubernetes Orchestration and Serverless GPU](./chapter_22_deployment/07_k8s_serverless.md)
  - [22.8 Long-Running Task Queues and Cost Governance](./chapter_22_deployment/08_task_queue_cost.md)

---

- [Part VI: Capstone Projects](./part6.md)

- [Chapter 23: Capstone Project: AI Coding Assistant](./chapter_23_coding_agent/README.md)
  - [23.1 Project Architecture Design](./chapter_23_coding_agent/01_architecture.md)
  - [23.2 Code Understanding and Analysis](./chapter_23_coding_agent/02_code_understanding.md)
  - [23.3 Code Generation and Modification](./chapter_23_coding_agent/03_code_generation.md)
  - [23.4 Test Generation and Bug Fixing](./chapter_23_coding_agent/04_testing_debugging.md)
  - [23.5 Full Project Implementation](./chapter_23_coding_agent/05_full_implementation.md)

- [Chapter 24: Capstone Project: Intelligent Data Analysis Agent](./chapter_24_data_agent/README.md)
  - [24.1 Requirements Analysis and Architecture Design](./chapter_24_data_agent/01_requirements.md)
  - [24.2 Data Connection and Querying](./chapter_24_data_agent/02_data_connection.md)
  - [24.3 Automated Analysis and Visualization](./chapter_24_data_agent/03_analysis_visualization.md)
  - [24.4 Report Generation and Export](./chapter_24_data_agent/04_report_generation.md)
  - [24.5 Full Project Implementation](./chapter_24_data_agent/05_full_implementation.md)

- [Chapter 25: Capstone Project: Multimodal Agent](./chapter_25_multimodal/README.md)
  - [25.1 Multimodal Capabilities Overview](./chapter_25_multimodal/01_multimodal_overview.md)
  - [25.2 Image Understanding and Generation](./chapter_25_multimodal/02_image_understanding.md)
  - [25.3 Voice Interaction Integration](./chapter_25_multimodal/03_voice_interaction.md)
  - [25.4 Practice: Multimodal Personal Assistant](./chapter_25_multimodal/04_practice_multimodal_assistant.md)
  - [25.5 Computer Use and GUI Agents](./chapter_25_multimodal/05_computer_use_agent.md)
  - [25.6 Video Understanding and Multimodal RAG](./chapter_25_multimodal/06_video_and_multimodal_rag.md)


---

# Appendix

- [Appendix A: Common Prompt Template Collection](./appendix/prompt_templates.md)
- [Appendix B: Agent Development FAQ](./appendix/faq.md)
- [Appendix C: Recommended Learning Resources and Communities](./appendix/resources.md)
- [Appendix D: Glossary](./appendix/glossary.md)
- [Appendix E: KL Divergence Explained](./appendix/kl_divergence.md)
- [Appendix F: Development Environment Setup](./chapter_setup/README.md)
  - [F.1 Python Environment and Dependency Management](./chapter_setup/01_python_setup.md)
  - [F.2 Installing Key Libraries](./chapter_setup/02_install_libs.md)
  - [F.3 API Key Management and Security Best Practices](./chapter_setup/03_api_key_management.md)
  - [F.4 Your First Agent: Hello Agent!](./chapter_setup/04_hello_agent.md)
