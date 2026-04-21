# Claude-Opus-4.7 发布一天，我把它变成了我的"知识库馆长"

> 不是让它写代码，也不是让它总结论文。
>我让它做了一件很朴素(这个想法来源于大神卡帕西，也看了一些其他人做的 agent,其他都做的也很好，但是感觉上手难度有点高)  **帮我把乱成一团的笔记、收藏、零散想法，自动归档成一本本能翻的书。**
>
> 花了大半天，写了个 Skill，叫 **Knowledge Vault**。
> 这篇讲清楚三件事：**为什么要做、它长什么样、怎么在 Claude Code / Codex / WorkBuddy 里直接用。**

目前我的知识库： 全书github地址： https://github.com/Haozhe-Xing/agent_learning
在线阅读地址：https://haozhe-xing.github.io/agent_learning/

大神karpathy,llm-wiki: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
其他人开源的 llm-wiki: https://github.com/nashsu/llm_wiki
---

## 一、先说结论：我需要的从来不是"AI 笔记"

过去三年我大概试过七八种知识管理工具：Notion、Obsidian、飞书文档、Mem、Reflect、Heptabase……

每一个都是：前两周很爽 → 一个月后变垃圾堆 → 半年后打不开。

真正的问题其实一句话就能说清：

> **AI Agent 时代，知识库不是给人看的，是给 Agent 读的。**
> 但我又不能放弃"人能翻得明白"——一本乱七八糟的书，Agent 也读不通。

所以我要的东西很简单：

1. 存储格式是**纯 Markdown**，Git 能管、AI 能读、断网能看；
2. 结构化成一**本书**，有目录、有章节、能一键 serve 成网页；
3. Agent 能**自己判断内容该归哪儿**，写进去并同步目录；
4. **不绑任何平台**——Claude Code 用得上，换 Codex 明天照样用。

答案非常朴素：**mdbook + 一个 Agent Skill**。

没有向量库、没有 embedding、没有知识图谱。
**只有文件、Markdown、和一个 `config.yaml` 索引。**

Claude-Opus-4.7 刚发布那天，我一口气把它跑通了。

---

## 二、它长什么样

每个知识库 = 一本 mdbook：

```
xiaohongshu-ops/
├── book.toml              # mdbook 配置
└── src/
    ├── SUMMARY.md         # 章节目录（Agent 负责同步）
    ├── README.md
    ├── 01-账号定位/
    ├── 02-内容创作/
    ├── 03-算法流量/
    ├── 04-涨粉互动/
    ├── 05-变现商业化/
    ├── 06-数据分析/
    └── 07-避坑指南/
```

一条 `mdbook serve`，浏览器里就是一本 GitBook。

我目前的"书架"长这样：

| ID | 类型 | 主题 |
|---|---|---|
| agent-learning | mdbook | AI Agent 开发（LLM / RAG / MCP / 多 Agent） |
| cpp-usaco-book | mdbook | C++ 竞赛编程（USACO Bronze→Gold） |
| usaaio-book | mdbook | AI 奥赛备考（ML / DL / Transformer） |
| xiaohongshu-ops | mdbook | 小红书运营（7 章、35 篇） |
| usaco-cpp-videos | video-collection | 教学视频成片 |
| learning-project | project-collection | 项目代码归档 |

整个系统的"大脑"就是一个 `config.yaml`：

```yaml
root: /Users/xxx/workspace/ft_send
knowledge_bases:
  - id: agent-learning
    path: agent_learning
    src_dir: src/zh
    scope: "AI Agent 开发教程：LLM、规划、记忆、工具、RAG、MCP..."
    tags: [ai, agent, llm, rag, mcp, langchain, multi-agent]
  - id: xiaohongshu-ops
    ...
```

**Agent 每次操作前先读这个文件，不凭记忆乱猜。** 这是整个 Skill 能稳定工作的地基。

---

## 三、Skill 只做三件事

### 1. Ingest：把一段内容归进某本书

流程很直白：

```
用户丢内容
  → Agent 读 config.yaml 拿到所有知识库
  → 提取 [标题 / 摘要 / 标签]
  → 对每本书打分：
      标签重合 40% + scope 语义 40% + 类型亲和 20%
  → score ≥ 60 直接归入 / 40~60 问一下 / <40 建议新建一本书
  → 写进 {chapter}/{slug}.md
  → 同步 SUMMARY.md
```

**决策过程会直接打印出来**，绝不是黑盒：

```
🧠 分类决策
标签: [ai, agent, rag]
候选:
- agent-learning → 85%（标签重合 3/3，scope 高度吻合）
- usaaio-book    → 35%（仅 ai 标签重合）
归入: agent-learning
```

你随时能按一下暂停。

### 2. Create KB：30 秒起一本新书

说一句"建一本关于量化交易研究的书"，Agent 会：

- 建好 `book.toml` + `SUMMARY.md` + `README.md`
- 在 `config.yaml` 追加一条记录
- 告诉你 `mdbook serve xxx` 就能看

### 3. Query / List / Lint

- **Query**：按标签筛候选库 → grep 搜 → 聚合答案、标来源；
- **List**：打一张表格看所有书；
- **Lint**：检查死链、孤儿文件、`book.toml.src` 跟 `src_dir` 对不对得上——**这步救了我好几次**。

就这三件事。

很多人做"AI 知识库"喜欢堆向量库、embedding、RAG……我刻意全砍了。因为这套"愚蠢"设计有两个好处没人能抢走：

1. **完全可移植**：换 Agent 平台，把 `.md` 文件夹拷过去就行，**零 lock-in**；
2. **人类友好**：不装软件能 `cat SUMMARY.md`；装个 mdbook 就是网页；丢 GitHub 就是 GitHub Pages。

---

## 四、为什么非得等 Claude-Opus-4.7

说实话，这套想法去年就有了。跑不通的原因很具体：

归档本质是个**多步决策**——

> 读索引 → 读内容 → 提要素 → 对 N 本书打分 → 找匹配章节 → 写文件 → 改 SUMMARY.md → 反馈用户。

中间任何一步出错都会崩。

**老版本模型最常犯的错，是在最后一步偷懒**：写完文件就返回成功，`SUMMARY.md` 不同步，mdbook 渲染不出来，下次 Agent 来看目录里没有这条，又给你重复写一份。

Claude-Opus-4.7 这一版我盯着跑了十几轮，两个差异很明显：

1. **它会主动回来改 SUMMARY.md**，连 `./` 相对路径前缀都处理对；
2. **长链条不掉线**：从读 config 到打分到落盘到同步目录，一个 prompt 里跑完，中途不会走偏去问你"要不要我继续"。

**这两件事看着小，但没做对之前，这个 Skill 根本不能用。**

更重要的是，Skill 生态在过去几个月终于**收敛**了——Claude Code、Codex、WorkBuddy 现在吃同一种格式：

```
skill-name/
├── SKILL.md              # 主提示词（工作流、约束、模板）
└── references/
    └── config.yaml       # 数据
```

**写一次，到处能跑。** 这比什么花里胡哨的 MCP 协议、Agent 框架都来得实在。

Knowledge Vault 不依赖任何平台 API，只依赖两件事：
- Agent 能读写本地文件；
- Agent 能按自然语言指令跑多步流程。

4.7 级别的模型，这两件事都够稳了。**窗口期就是现在。**

---

## 五、三个真实用过的场景

### 场景 1：刷到一篇讲 Agentic RAG 的文章

我扔给 Agent：

> "把这个 URL 归档一下：https://xxxxx"

它回：

```
🧠 分类决策
候选:
- agent-learning → 92%（3/4 标签重合，scope 高度吻合 RAG）
- usaaio-book    → 28%
归入: agent-learning → rag/agentic-rag-introduction.md

✅ 已归档
🆕 agent_learning/src/zh/rag/agentic-rag-introduction.md
✏️ agent_learning/src/zh/SUMMARY.md（新增目录项）
```

**整个过程 15 秒。我只输入了一句话。**

### 场景 2：零散笔记不再死在草稿箱

做小红书知识库时，我经常一边刷案例一边记零散笔记。以前这些笔记最后全死在 Notes.app 里。

现在：

> "记一下：小红书违禁词里最新又加了『绝对化用语 + 医疗疗效词』组合会直接限流，案例是 XXX。"

Agent 自动归到 `07-避坑指南/违禁词治理.md`，目录同步，下次搜"违禁词"一抓就有。

最爽的一点是：**半年后我换电脑、换 Agent 平台，只要把文件夹同步过去，一切都还在。**

### 场景 3：30 秒起一本新书架

> "给我建一本关于量化交易研究的书，叫 quant-research，tag 里加 python、backtest、factor。"

30 秒后 `book.toml` / `SUMMARY.md` / `README.md` 全备好，`config.yaml` 登记好。

---

## 六、怎么跑起来

### 方式 A：Claude Code / Codex CLI

把 Skill 丢到 `~/.claude/skills/` 或 `~/.codex/skills/`，改一下 `references/config.yaml` 里的 `root` 指向你自己的知识库根目录。

对话里说"归档到知识库"、"整理一下这个"——Skill 自动被触发。

### 方式 B：WorkBuddy（我主力用法）

丢到 `~/.workbuddy/skills/knowledge-vault/`，IDE 内直接 `@skill://knowledge-vault` 调用。

### 方式 C：任意 Agent 框架

`SKILL.md` 本身就是一段**纯提示词 + 工作流**。任何能"读文件 + 写文件 + 循环判断"的 Agent 都能跑。

我实测过 Claude 3.7 / 4 / 4.7、GPT-4.1、Gemini 2.5。**4.7 是目前唯一不用我中途干预就能完整跑完的模型**，尤其在"同步 SUMMARY.md"这步。

---

## 七、两句想说的话

给做 AI 产品的朋友：

> **不要再造"AI 版 Notion"了。**
> 真正稀缺的不是更炫的编辑器，是"Agent 可操作的知识结构"。
> Markdown + mdbook 这种老技术栈，在 Agent 时代反而是最优解——因为它**足够笨**，笨到 Agent 可以完全接管。

给被"AI 笔记产品"轰炸的用户：

> 别再找"功能最多"的工具了。
> 找"**你哪天离开它，能把所有数据带走**"的工具。
> Knowledge Vault 没有数据库、没有云同步、没有厂商绑定。它就是你电脑上的一堆 `.md`。
> 哪天它不好用了，你删掉 Skill，笔记还是笔记。

---

## 八、Roadmap & 开源

在做的：

- [ ] **自动 Lint + 一键修复**
- [ ] **跨书交叉链接**（一个概念在多本书出现时自动挂 related）
- [ ] **章节合并 / 拆分辅助**（长到要拆章时让 Agent 提议）
- [ ] **与 Claude Code Hooks / Codex 工作流的更深集成**

Skill 会整体开源。如果你想提前用，或者想一起折腾 Agent Skill 生态，评论区或私信都行。


---

**一句话收尾**：

> Claude-Opus-4.7 发布一天，最让我兴奋的不是它写代码更猛了——
> 是我第一次可以**用一句自然语言，把自己的知识组织成一本能翻的书**。
>
> 这件事，我觉得比写代码酷多了。

