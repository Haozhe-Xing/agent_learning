# 17.7 总结：六大 Harness 框架选型矩阵

> 🐋 *"一张图选完整个 Agent 工具栈。"*

---

## 一、本节目标

读完 14–17 章，你已经看过四个开源 Harness / 框架 + 12–13 章的两个代码框架。本节把这六个项目汇总成一张决策表，让你未来面对"用谁"的问题时，能在 5 分钟内得出答案。

---

## 二、第三部分全景回顾

```
第三部分：框架实战篇（6 章）
│
├─ 代码框架（让你"造"Agent）
│   ├─ 12 章 LangChain —— 装配库，链 / 工具 / Agent / LCEL
│   └─ 13 章 LangGraph —— 有状态图，检查点 / 条件路由 / Human-in-the-Loop
│
└─ 行动型 Harness（让你"用"Agent）
    ├─ 14 章 OpenClaw      —— 跨平台个人助理（多渠道）
    ├─ 15 章 Hermes Agent   —— 自我进化的私人助理（自进化）
    ├─ 16 章 Claude Code    —— 工业 IDE 范本（编程工具）
    └─ 17 章 DeepSeek Harness —— 一切皆插件的开源底座（自建平台）
```

---

## 三、选型矩阵（一张表）

### 3.1 横轴：你"想要什么"

| 你的需求 | 最适合 | 次选 |
|---------|--------|------|
| **快速接入多模型/多工具** | LangChain | OpenClaw |
| **复杂有状态 + 人机协作** | LangGraph | LangChain + LCEL |
| **Agent 住进 WhatsApp/Telegram** | OpenClaw | Hermes Agent |
| **Agent 自动学新技能（自进化）** | Hermes Agent | （尚无第二家）|
| **编程 IDE 中跑 Agent** | Claude Code | DeepSeek Harness |
| **搭一个长期可控的 Agent 平台** | DeepSeek Harness | OpenClaw + 自研 |
| **教学 / 训练 / 学习原理** | Claude Code（源码解剖）+ LangChain（库） | OpenClaw |
| **完全避开商业产品** | DeepSeek Harness + OpenClaw + Hermes 三件套 | 自研 |
| **生产级 SaaS Agent** | LangChain + LangGraph + LangSmith | OpenClaw（需配大量 ops 工具） |
| **科研 / 模型基准** | LangChain + 最小化整合 | DeepSeek Harness --profile minimal |

### 3.2 选型决策树

```
Q1: 你想立刻"用上"一个 Agent，还是想"自己搭"？
├─ 立刻用 → 行动型 Harness 子树
│   ├─ 把 Agent 放聊天 App？→ OpenClaw（第14章）
│   ├─ 让 Agent 越长越聪明？→ Hermes（第15章）
│   ├─ 软件工程 IDE 工作？→ Claude Code（第16章）
│   └─ 搭长期可控的平台？→ DeepSeek Harness（第17章）
│
└─ 自己搭 → 代码框架子树
    ├─ 需要复杂流程 + 状态？→ LangGraph（第13章）
    ├─ 需要快速接入多工具？→ LangChain（第12章）
    └─ 自己造轮子？→ 读 6 章任一"借鉴"小节，结合第 8 章 Harness Engineering
```

---

## 四、6 框架对照表（一图全览）

| 维度 | LangChain | LangGraph | OpenClaw | Hermes | Claude Code | DeepSeek Harness |
|------|-----------|-----------|----------|--------|-------------|------------------|
| **类型** | 代码框架 | 代码框架 | Harness | Harness | Harness | Harness |
| **许可** | MIT | MIT | MIT | MIT | source-avail | MIT |
| **核心范式** | Runnable 链 | 状态图 | 多渠道 | 自进化 | 工业 IDE | 一切皆插件 |
| **可换内核** | — | — | ❌ | ❌ | ❌ | ✅ |
| **模型无关** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| **可定制深度** | 高（代码） | 高（代码） | 中（Skill） | 中（Skill） | 低（Config） | 极高（插件） |
| **上手门槛** | 中 | 中 | 极低 | 极低 | 低 | 中 |
| **Skill 协议** | Tool ABC | Tool ABC | SKILL.md | SKILL.md | SKILL.md | SKILL.md |
| **生产就绪度** | 高 | 高 | 高 | 高 | 高 | 预览版 |
| **典型 30 天使用 ROI** | 中 | 高 | 高（消费级） | 高（自进化） | 高（开发） | 中（搭平台） |

---

## 五、按"长期价值"排序

一个**真正会用 Agent 的人**，工具栈可能在 1 年内经历这样的演化：

```
第 1 个月 ─► LangChain：从零搭 Agent（学原理）
  │
第 2-3 个月 ─► LangGraph：上状态、循环、人机协作（学工程）
  │
第 4-6 个月 ─► Claude Code：日常 IDE 伴侣（高效开发）
  │
第 6-12 个月 ─► OpenClaw + Hermes：个人助理接管日常事务
  │
        同时 ─► DeepSeek Harness：搭建长期可控的开源底座
```

**注意**：这不是"必须按此顺序"，而是"收益曲线如此自然展开"——你可以跳过任何一步。

---

## 六、三种典型技术栈推荐

### 6.1 学术研究 / 教学型

```
LangChain（库）
  + LangGraph（有状态）
  + 本地 LLM（Ollama）
  + 阅读 Claude Code 16.4 事件后源码分析作为范例
```

- 上手快；
- 完全开源 + 模型可控；
- 教学价值最高（每个抽象可读可改）。

### 6.2 消费级 / 个人助理型

```
OpenClaw（聊天渠道）
  + Hermes Agent（自进化）
  + 任意 LLM（云端或本地）
  + Docker 沙箱部署
```

- 上手极低；
- 多渠道 + 自进化 = "会陪着你长大的 Agent"；
- 长期使用 ROI 最高。

### 6.3 工程团队 / 生产级

```
Claude Code（编程 IDE，源/原料）
  + LangChain + LangGraph（生产级服务编排）
  + MCP 服务器生态（外接工具）
  + LangSmith（可观测性）
```

- 工业级稳定性（商业团队背书）；
- 生态最完整；
- 适合中大型团队的工程化场景。

### 6.4 长期自主 / "不想被卡脖子"型

```
DeepSeek Harness（开源底座）
  + OpenClaw/Hermes/Claude Code 任选作为"上层 UI"
  + 自研插件生态
  + 模型 fallback 链
```

- 完全可控；
- 模型无关；
- 适合"长期平台"型项目。

---

## 七、第三部分小结

我们用了 6 章把整个"行动型 Agent"图景铺开：

1. **认知层**——分清"代码框架 vs 行动型 Harness"；
2. **库层**——LangChain + LangGraph 是"让你造 Agent 的库"；
3. **产品层**——OpenClaw / Hermes / Claude Code / DeepSeek Harness 是"让你用 Agent 的产品"；
4. **范式层**——它们其实只是同一组（事件流 + 沙箱 + 状态图 + 工具 + 权限）的不同实现；
5. **工程层**——读完任何一个，都能"看清其他三个"。

---

## 八、接下来

第三部分到此结束。进入**第四部分：多 Agent 系统篇（第18章起）**——讨论多个 Agent 怎么协作、通信、协议化；为更大规模的复杂任务做准备。

> - **第18章 多 Agent 协作** —— Supervisor / 去中心化 / CrewAI 实战
> - **第19章 Agent 通信协议** —— MCP / A2A / ANP
> - **第20章 Agent 的评估与优化**
> - **第21章 安全与可靠性**
> - **第22章 部署与生产化**
> - **第23章 项目实战：AI 编程助手**
> - **第24章 项目实战：智能数据分析 Agent**
> - **第25章 项目实战：多模态 Agent**

---

## 九、本节小结

| 主题 | 关键要点 |
|------|---------|
| 6 框架全景 | LangChain / LangGraph / OpenClaw / Hermes / Claude Code / DeepSeek Harness |
| 选型决策 | 用 vs 搭 + 场景定位 |
| 长期价值 | 6 框架在不同阶段各有最佳 ROI |
| 技术栈推荐 | 学术 / 消费 / 工程 / 长期自主——四种场景四种栈 |
| 接下来 | 第四部分：多 Agent 系统篇 |

---

*返回章节首页：[第17章 DeepSeek Harness：一切皆插件的开源底座](./README.md)*
