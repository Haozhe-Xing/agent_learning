# 论文来源与结论核对台账

核验日及本轮选文截止日：2026-10-22。正文对象：两语4.6。核题名、作者、版本和下列原文章节；不代表复现论文、核验当前产品或审结全书研究综述。正文的实践建议是本书设计，不冒充论文实验结论。

| 作品与作者 | 固定来源 | 本次核读范围 |
|---|---|---|
| Generative Agents: Interactive Simulacra of Human Behavior — Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein | [2304.03442 v2，2023-08-06](https://arxiv.org/abs/2304.03442v2)；首次提交2023-04-07 | [正文§4、§6及§7开头的相关段落](https://arxiv.org/html/2304.03442v2)：记忆流/检索/反思/规划；受访、消融及记忆错误 |
| MemGPT: Towards LLMs as Operating Systems — Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez | [2310.08560 v2，2024-02-12](https://arxiv.org/abs/2310.08560v2)；首次提交2023-10-12 | [正文§2–3](https://arxiv.org/html/2310.08560v2)：主/外部上下文、队列和函数控制流、长对话与文档实验 |
| MemoryBank: Enhancing Large Language Models with Long-Term Memory — Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, Yanlin Wang | [2305.10250 v3，2023-05-21](https://arxiv.org/abs/2305.10250v3)；首次提交2023-05-17 | [正文§2、§4](https://arxiv.org/html/2305.10250v3)：分层存储、检索、简化衰减/巩固、真实/模拟对话实验 |
| Cognitive Architectures for Language Agents — Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, Thomas L. Griffiths | [2309.02427 v3，2024-03-15](https://arxiv.org/abs/2309.02427v3)；首次提交2023-09-05，TMLR版 | [正文§4–6相关段落](https://arxiv.org/html/2309.02427v3)：工作/长期记忆、内部/外部动作、案例、决策及设计边界 |
| HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models — Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, Yu Su | [2405.14831 v3，2025-01-14](https://arxiv.org/abs/2405.14831v3)；首次提交2024-05-23，NeurIPS 2024 | [正文§2–3](https://arxiv.org/html/2405.14831v3)：OpenIE图索引、实体链接、PPR和多跳QA实验 |
| Zep: A Temporal Knowledge Graph Architecture for Agent Memory — Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef | [2501.13956 v1，2025-01-20](https://arxiv.org/abs/2501.13956v1) | [正文§2–4](https://arxiv.org/html/2501.13956v1)：三层图、双时间线、检索及DMR/LongMemEval实验 |
| LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory — Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, Dong Yu | [ICLR 2025正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf)，以会议PDF为本节固定文本 | 正文§3–5：500题、五种能力/七类题型，历史构造，indexing/retrieval/reading及评估设置 |
| C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks — Xueshu Chen, Yan Wang, Zihao Xue, Jiefu Li, Zhenfang Liu, Jayden Chen, Zhen Bi, Jungang Lou | [2609.29735 v1，2026-09-24](https://arxiv.org/abs/2609.29735v1) | [正文§2–3](https://arxiv.org/html/2609.29735v1)：RAMU/BMER、源指针与多维读取预算；MemLens子集设置/结果，未复现 |

## 已回写的具体修正

- 删除“最有影响力”“黄金范式”“图谱必然优于向量”“理论无限记忆”和未经核验的当前产品排名；不再把不同任务与版本串成单一优胜谱系。
- MemGPT区分主上下文的三个区段和上下文之外的存储；外部内容须显式调入，不是仍在上下文内的另一区域。撤下无来源的mem0归属说法。
- CoALA补程序性记忆，并将记忆、动作、决策分开，避免把四种动作画成固定执行流水线。
- MemoryBank的`R=exp(-t/S)`是其简化模型，不另造“低于阈值必归档”的API；衰减不等于用户撤销或物理删除。
- 保留作者所用任务/版本和适用边界，正文不转述跨模型价格、时延或当前榜单。C3M以预印本选读处理，不因较新即作必学依赖。
- 两语采用同一8篇、同一读法、同一自建六项练习；练习不冒充LongMemEval或任何论文复现。3幅示意图为本书重绘，标明所据版本。

## 材料保留与未完成范围

原4.6两语全文及SHA-256见 `editorial_archive/2026-10-22/memory_readings/MANIFEST.md`，在发布源之外保留。未选条目尚未逐篇裁定，不能把“未保留”解读为“不存在”。原图文件未删除，其版权和来源完整性须单独审校。

PUB-07中16.6、19.6、24.7等论文页仍待同样的逐篇核验与双语取舍；4.6完成不外推为全书完成。这里不覆盖产品SDK/安装兼容性、论文开源代码许可证或研究成果独立复现。
