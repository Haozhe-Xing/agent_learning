# 多 Agent 与安全论文来源核对台账

核验日/选文截止日：2026-10-22。对象为两语16.6与19.6。以下逐项通过web读取arXiv版本页和固定版HTML的所列方法、实验及限制段落；不是实验复现、当前SDK文档、全库文献审结或漏洞扫描。作者列表按版本元数据记录，正文使用首作者加“等”。

## 16.6：七篇，双语同一清单

| 固定作品、作者 | 来源 | 实际核读与正文保留的边界 |
|---|---|---|
| **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework** — Sirui Hong, Mingchen Zhuge, Jiaqi Chen, Xiawu Zheng, Yuheng Cheng, Ceyao Zhang, Jinlin Wang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, Jürgen Schmidhuber | [元数据：2308.00352v7，2024-11-01](https://arxiv.org/abs/2308.00352v7)，首次2023-08-01；[正文](https://arxiv.org/html/2308.00352v7) | §3.1–3.3的五种角色、SOP、结构工件、消息池/订阅与执行反馈；§4.1–4.4的HumanEval/MBPP/SoftwareDev、指标和消融。SoftwareDev为70题，文中比较抽七题；可执行性为分级评价，不能改写成通用百分比成功率或把收益单因果归于通信格式。 |
| **ChatDev: Communicative Agents for Software Development** — Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, Maosong Sun | [2307.07924v5，2024-06-05](https://arxiv.org/abs/2307.07924v5)，首次2023-07-16；[正文](https://arxiv.org/html/2307.07924v5) | §3的设计/编码/测试三阶段、五子任务及指导者/助手澄清；§4的SRDD 1200题、完整性/可执行性/一致性/质量定义、ChatGPT-3.5与轮数设置；§6限制。去除旧稿四阶段断言及无条件O(N²)→O(N)结论。没有占位或能启动不等于业务合同通过。 |
| **AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation** — Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, Chi Wang | [2308.08155v2，2023-10-03](https://arxiv.org/abs/2308.08155v2)，首次2023-08-16；[正文](https://arxiv.org/html/2308.08155v2) | §2可对话主体与conversation programming；§3六应用和§4讨论，并读附录D相关配置段。数学、QA、ALFWorld等为不同设置，非统一受控排名；历史类名不当作当前API，工具执行不构成隔离或身份认证。 |
| **Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks** — Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang (Eric) Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, Saleema Amershi | [2411.04468v1，2024-11-07](https://arxiv.org/abs/2411.04468v1)；[正文](https://arxiv.org/html/2411.04468v1) | §4外/内循环、task/progress ledger及四执行主体；§5.1–5.4的重置、重复、GAIA/AssistantBench/WebArena、消融/错误分析，§6限制/风险。表1的人类与系统分数不支持旧稿“接近人类”；不保留“30%成功来自重规划”或AutoGen0.4归属断言。模型变体、拒绝和任务集条件保留在原文，不生成当前排名。 |
| **MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents** — Kunlun Zhu, Hongyi Du, Zhaochen Hong, Xiaocheng Yang, Shuyi Guo, Zhe Wang, Zhenhailong Wang, Cheng Qian, Xiangru Tang, Heng Ji, Jiaxuan You | [2503.01935v1，2025-03-03](https://arxiv.org/abs/2503.01935v1)；[正文](https://arxiv.org/html/2503.01935v1) | §3.1–3.3的MARBLE、六场景、四拓扑、里程碑/任务/协作指标；§4–5的模型、迭代和数量消融。LLM里程碑检测/评分与规则结果分开，不能将某场景图结构表现外推为通用最优拓扑。 |
| **Why Do Multi-Agent LLM Systems Fail?** — Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica | [2503.13657v3，2025-10-26](https://arxiv.org/abs/2503.13657v3)，首次2025-03-17；[正文](https://arxiv.org/html/2503.13657v3) | §3初始150轨迹、人工标注与LLM扩标，1642轨迹/七系统；§4三类14模式；§5分析及图4“非跨系统性能比较”说明。采样标签非生态发生率，分类非穷尽，标注不构成单一因果责任证明。 |
| **Rethinking Multi-Agent Collaboration: When More Is Less** — Yishuo Yuan, Yibo Wu, Yihan Zhang, Minyuan Sun, Shenliang Li, Xinkai Ma, Yifan Li, Jiaheng Liu | [2609.19759v2，2026-09-18](https://arxiv.org/abs/2609.19759v2)，首次2026-09-17；[正文](https://arxiv.org/html/2609.19759v2) | §3依赖轨迹/bridge edges，§4按需生成与消息获取，§5四基准与消融。原文主实验统一Codex CLI harness/DeepSeek-V4-Pro、总token合计、最多4主体/子任务深度1；所读结果随依赖结构变化，不将这些参数推荐为通用人数，不把预印本当成熟生产依赖。 |

## 19.6：八篇，双语同一清单

| 固定作品、作者 | 来源 | 实际核读与正文保留的边界 |
|---|---|---|
| **Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection** — Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz | [2302.12173v2，2023-05-05](https://arxiv.org/abs/2302.12173v2)，首次2023-02-23；[正文](https://arxiv.org/html/2302.12173v2) | §3注入投递与影响分类；§4的当时Bing Chat/Copilot/合成应用演示；§5限制与复现讨论。文字不是OS权限，工具副作用依赖宿主暴露能力；案例不估计生态发生率，也不核当前产品漏洞。 |
| **InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents** — Qiusi Zhan, Zhixiang Liang, Zifan Ying, Daniel Kang | [2403.02691v3，2024-08-04](https://arxiv.org/abs/2403.02691v3)，首次2024-03-05；[正文](https://arxiv.org/html/2403.02691v3) | §2的17用户场景×62攻击场景=1054例、返回模板、假定用户工具已正确调用和后续模拟；§3–4的设置/结果及§8限制。基本/增强固定提示、有效响应/全部样例分母须说明；不是任意真实服务、完整权限或长轨迹覆盖。 |
| **AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents** — Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr | [2406.13352v3，2024-11-24](https://arxiv.org/abs/2406.13352v3)，首次2024-06-19；[正文](https://arxiv.org/html/2406.13352v3) | §3环境状态、用户/攻击任务、97任务/629用例；§3.4的benign utility、utility under attack（无对抗副作用）及targeted ASR；§4基线/消融/防御。无攻击成功不等于原始任务完成，静态攻击失败不排除适应性攻击。 |
| **StruQ: Defending Against Prompt Injection with Structured Queries** — Sizhe Chen, Julien Piet, Chawin Sitawarin, David Wagner | [2402.06363v2，2024-09-25](https://arxiv.org/abs/2402.06363v2)，首次2024-02-09；[正文](https://arxiv.org/html/2402.06363v2) | §4.1–4.4的保留分隔token/数据侧处理和结构化指令训练；§5效用AlpacaEval、指定输出攻击、自适应攻击及前端/训练消融；§6讨论。普通XML/JSON不等于StruQ；模型鲁棒性不授予工具权限，不与SecAlign混成同一训练算法。 |
| **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation** — Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Wei Koh, Mohit Iyyer, Luke Zettlemoyer, Hannaneh Hajishirzi | [2305.14251v2，2023-10-11](https://arxiv.org/abs/2305.14251v2)，首次2023-05-23；[正文](https://arxiv.org/html/2305.14251v2) | §3的原子事实与给定知识源支持比例、人物传记/Wikipedia实验；§4的自动估计器与Limitations明确不测factual recall。支撑精度不等于需求覆盖；自动估计非每条完美，冲突/争议是限制，6.5摘录检查不冒充该估计器。 |
| **SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models** — Potsawee Manakul, Adian Liusie, Mark J. F. Gales | [2303.08896v3，2023-10-11](https://arxiv.org/abs/2303.08896v3)，首次2023-03-15；[正文](https://arxiv.org/html/2303.08896v3) | §5五类变体、§6的238 WikiBio/GPT-3生成篇章与句子标注，§7检测/排序。zero-resource为无外部事实库，不是零调用；一致性不能排除一致地错。撤下旧稿用同一图冒充Self-Consistency、多数票即事实验证的表述。 |
| **Loopjacking: Hijacking Human-in-the-Loop Approval** — Adithyan Arun Kumar | [2609.21081v1，2026-09-17](https://arxiv.org/abs/2609.21081v1)；[正文](https://arxiv.org/html/2609.21081v1) | §2两变体/必要前提、§3–4产品路径和正负控制，§6完整操作绑定/执行前重查，§7限制。LangGraph为条件化的共享thread/in-memory Agent Server构成，非所有LangGraph；Agno为所列7版本而非连续全版本；OpenAI Agents SDK 0.22.0/0.22.2为本研究负对照，非“永久免疫”。正文不提供当前版本漏洞结论或生态流行率。 |
| **Persistent Billable State: Denial-of-Wallet Attacks and Defenses in Tool-Calling LLM Agents** — Jinqian Zhang, Haojun Xia, Shujiang Wu, Jingkun Yue, Xia Zhang, Zhangpei Cheng, Bibo Tu | [2609.28585v1，2026-09-23](https://arxiv.org/abs/2609.28585v1)；[正文](https://arxiv.org/html/2609.28585v1) | §4攻击面、§5测量/历史政策、§6核心243执行/六家族；§7的D1输入/D2增长/D3递归/D4累计计费暴露，§8冻结回放和限定在线工作流。区分回放、在线实验和3830仓库的code-visible proxy扫描单位；触发轮已发生费用不能追回，不转述费率或当作通用阈值/安全率。 |

## 双语、图与练习的编辑合同

- 两语各为七/八篇，相同排序、方法/边界/迁移段与六/八项自建练习。不是把目录编号调整计为事实审校。论文数不是全面覆盖声明；未纳入的安全/隐私主题由主线章节继续承担。
- 16.6练习要求引用指定版本并正确求同币值之和；基线A-v1=40 CNY、B-v1=60 CNY独立预期100，另立A-v1/B-v2=80 CNY的输入快照并事先冻结120预期。不是运行后按候选改写100。覆盖缺资料、异币种、版本更新、重复来源与失败。没有将该资料任务冒充16.5开发团队已实现功能；既有状态只用于对照。
- 19.6练习分开正常效用、攻击效果、支撑、覆盖、审批绑定、累积预算和验证器故障；6.5/11.5只复用其实际提供的局部边界，不称为完整安全系统。没有产生基准分数、实测防御提升或镜像测试。
- 本轮正文仅保留并重绘五种图、两语共10张：`chapter_18_multi_agent_06_{timeline,magentic_one}.svg`；`chapter_21_security_06_{injection_attack,struq_defense,factscore}.svg`。以本书原创文字和几何解释机制，不嵌入论文图、截图、商标或论文整段文字；SVG元数据标明固定来源和本书示意范围，沿仓库MIT许可。没有把arXiv允许托管误解为任意论文插图都可再版。
- 原稿其他PNG/SVG及作者材料未删除。它们当前未被这四篇发布正文引用；来源、图号、原作授权与翻译/改编许可仍需单独清查，不能由本轮新图覆盖。

## 原文保留与未核范围

修订前四原稿逐字节副本见 `editorial_archive/2026-10-22/multi_agent_readings/` 与 `security_readings/` 的MANIFEST；记录原路径、字节和SHA-256。归档是编辑材料，不能作为事实来源。未选条目尚未逐篇核验，不因移出即被判假，也不能从快照直接恢复为已核内容。

本轮核读限于上表方法、实验设置/结论边界及元数据；没有逐行审核研究代码、核验所有原文引用、安装研究框架、重跑论文实验、验证现有产品修复或核查当前报价。2026新文已读取其固定原文，仍以预印本选读处理。全书其他论文页和主线安全工程的就绪性由各自审校/验证记录负责。

## 本次编辑产物检查

- 四原稿的319661字节快照与MANIFEST SHA逐一复核；两语固定来源ID/版本/顺序相同。16.6各9个二级标题、14个表格行；19.6各10个二级标题、16个表格行，包含对应的六/八条自建练习。未留下不可执行Python或SDK片段。
- 10幅SVG均以本机已有sharp实际栅格化为1920×1000 PNG，并逐张目视检查，包括改成“经专门训练的模型/Trained model”的StruQ最终版。预览位于临时目录 `/tmp/agent-learning-collab-security-svg-review/`，不加入发布源；SVG XML可解析，未见溢出/遮挡。
- `/tmp/agent-learning-integration-venv/bin/python tools/check_book.py`：422 Markdown/687 SVG，0错；`tools/check_learning_path.py`：25章/7部分/两语，0错；`git diff --check`通过。文件总数为该次共享工作区快照，不代表全书内容完成出版审校。
- `/tmp/agent-learning-integration-venv/bin/python -m pytest examples/dev_team/tests/test_dev_team.py -q`：既有28项通过，仅复核16.6提及的开发团队生命周期入口。没有新增镜像测试、执行研究框架或报告论文复现分数；完整双语发布成品检查仍由最终构建记录负责。
