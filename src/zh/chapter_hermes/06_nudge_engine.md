# 15.6 Nudge Engine 与跨会话学习

> ☤ *"最聪明的 Agent 不是反应最快，而是会主动想'我学到了什么'的那个。"*

---

## 一、Nudge Engine 是什么

Hermes 的 Self-Evolving（15.4）是"被动"的——它只在每次任务完成后评估。Nudge Engine 是**主动**的：每隔一段时间，Hermes 会**自己**问自己：

- 我最近学到了什么值得沉淀的？
- 我对用户的理解有变化吗？
- 我能否把过去几天的某些经验总结成 Skill？
- 我今天的某些步骤是不是太累赘了？

这种"主动反思"让 Hermes 比纯被动的 Agent 高出一个量级。本节拆解它的实现。

---

## 二、Nudge 的 4 种触发机制

Nudge Engine 不是"每分钟问自己"——而是按以下 4 种触发条件：

| 触发类型 | 触发时机 | 触发的动作 |
|---------|---------|-----------|
| **时间触发** | 每 30 分钟 | 评估"是否有值得提炼的经验" |
| | 每 24 小时 | 评估"用户画像是否需要更新" |
| | 每周一次 | 跑一次"全 Skill 库评审" |
| **事件触发** | 会话结束 | 判断是否提炼 Skill |
| | 工具调用失败 ≥ 3 次 | 判断是否更新 Skill 或 USER.md |
| | 用户说"记住 XXX" | 写入 MEMORY.md |
| **阈值触发** | 会话条数 > 100 | 压缩远端上下文 |
| | 周末 | 跑一次"周复盘" |
| **外部触发** | CRON 任务（用户配置） | 跑特定 Skill |

下面依次看每种触发如何实现。

---

## 三、Nudge Loop 的实现（简化）

```python
# hermes/nudge/loop.py —— 简化
class NudgeLoop:
    def __init__(self, memory, self_evolving, honcho):
        self.memory = memory
        self.se = self_evolving
        self.honcho = honcho

    async def maybe_nudge(self, trigger: str):
        # 1. 检查触发是否生效（开关、频率、预算）
        if not self._enabled(trigger):
            return

        # 2. 调用对应的"反思 prompt"
        if trigger == "task_done":
            await self._maybe_extract_skill()
        elif trigger == "user_pref_changed":
            await self._update_user_model()
        elif trigger == "weekly_review":
            await self._weekly_skill_review()
        elif trigger == "session_long":
            await self._compress_session()

    async def _maybe_extract_skill(self):
        # 取最近一次会话
        session = await self.memory.get_recent_session(limit=1)
        # 评估是否值得提炼
        if self.se.should_extract(session.trajectory):
            skill_md = await self.se.extract_skill_md(session.trajectory)
            await self.se.save_draft(skill_md)
            print(f"📝 Hermes: 草稿 Skill 已生成 — {skill_md.name}")
```

### 3.1 运行示例：一次"任务完成"触发的 Nudge

假设用户刚让 Hermes 完成一次"整理收件箱"任务，任务结束时 `maybe_nudge("task_done")` 被触发：

```
[20:30] 用户任务"整理收件箱"完成
[20:30] Nudge 触发：task_done
[20:30] _maybe_extract_skill() 取最近会话轨迹（8 步，成功）
[20:30] should_extract() 评估：8 步 ≥ 5，成功 → True
[20:31] extract_skill_md() 调 LLM 提炼 → 生成 summarize-inbox 技能
[20:31] save_draft() 落盘为草稿（status: draft）
[20:31] 输出：📝 Hermes: 草稿 Skill 已生成 — summarize-inbox
```

**这个 trace 的关键点**：整个 Nudge 过程（从触发到落盘）**只花了 1 秒左右，而且是异步的**——它发生在任务完成之后，不阻塞用户和 Agent 的对话。用户只会看到"任务完成了"，不会感知到"Agent 又在后台提炼了一个技能"。

**为什么 Nudge 必须异步、必须离线？** 因为提炼要调一次 LLM（生成 SKILL.md），这是有成本、有延迟的操作。如果把它塞进"用户发消息 → Agent 回复"的热路径里，每次任务结束用户都要多等 1-2 秒。**离线 Nudge 把"反思"从热路径剥离，让 Agent 的"学习"对用户完全透明**。

---

## 四、Honcho Dialectic：用户建模

### 4.1 概念

**Honcho Dialectic** 是 Hermes 内置的"用户建模子系统"——它的工作是：

1. **持续观察**：把每一轮对话里的"用户偏好信号"收集起来；
2. **结构化输出**：把这些信号提炼成 `USER.md` 的一段结构化描述；
3. **写回**：把 `USER.md` 更新到磁盘。

> "Dialectic" 这个词来源于"对话式辩证"——它意味着 USER.md 不是一次性生成的，而是通过**多轮"自己对自己提问"**反复精炼的。

### 4.2 Honcho 的提问模板（简化）

```python
HONCHO_PROMPT = """
Based on the user's recent conversations, infer:
1. Communication style preferences (terse, detailed, formal, casual)
2. Working hours and time-of-day patterns
3. Common tools they use (Python vs JS, etc.)
4. Pain points they repeatedly mention
5. What they value (correctness, speed, brevity, deep detail)
6. What they avoid (emoji, marketing language, unverified claims)

Output as a structured JSON or Markdown table that could be added to USER.md.
"""
```

调用 LLM 后，Honcho 得到类似这样的输出：

```json
{
  "communication_style": "terse, structured, no emoji",
  "working_hours": "上午深度工作，下午会议，晚上写作",
  "tools_primary": ["Python", "Node", "TypeScript"],
  "pain_points": [
    "Agent 输出的'无法验证'数据（如 Star 数）总让 ta 警觉",
    "Ai 推送前后不一致"
  ],
  "values": ["correctness", "deep detail"],
  "avoids": ["emoji-heavy text", "marketing speak"]
}
```

### 4.3 USER.md 的更新是"增量的"

Hermes 不会**重写** USER.md，而是**增量更新**——每次只根据最新观察修一行，避免一次性重写丢东西：

```markdown
# USER.md —— 用户建模（增量维护）

## 基本画像
...

## 工作模式
...

## 偏好
- 简洁直接     [更新于 2026-06-10]
- 中文为主、英文术语 [更新于 2026-06-10]
- 提交前要确认破坏性动作  [更新于 2026-07-12]
- 不引用动态数字   [更新于 2026-08-03，由 Honcho 更新]
```

### 4.4 Honcho 的失败模式与抑制

Honcho 不应该无限制地"自我推测"——它有抑制机制：

```
抑制条件：
- 用户最近一周没说新信号 ─► 不更新
- 本次更新的 confidence < 0.6 ─► 不写入
- 与现有 USER.md 已有内容冲突 ─► 不写入（保留旧）
- 用户明确 say "忘了那个" ─► 删除对应行
```

---

## 五、周期性 Skill 评审（Weekly Review）

Nudge 还有一个重要的子流程：**每周 Skill 评审**——它把"系统能用度"的判断从"单次任务"提升到"长期使用"。

### 5.1 评审流程

```
每周一次 ─► 对 ~/.hermes/skills/ 下所有 Skill 跑"健康度评估"
              │
              ├─ 最近 7 天被调用次数
              ├─ 成功率
              ├─ 平均 token 消耗
              ├─ 用户是否给了显式 feedback
              │
              ▼
       生成"健康度报告"，分类为：
         ✅ 健康（继续保留）
         ⚠️ 需优化（触发 self-improve）
         ❌ 濒死（提示用户决定）


生成摘要 ─► 写入 MEMORY.md（长期）
           ─► 用户可选：邮件推送 / Telegram 推送
```

### 5.2 用户可选的"反馈信号"

Hermes 提供几个让用户**主动**反馈 Skill 表现的方式：

```
你:    "那个 book-flight skill 不好用，别再用了"
Hermes: ⚠️ 已标记 book-flight 为 "deprecated"，下次 self-improve 将跳过它

你:    "search-flights skill 应该返回价格区间，不是单点"
Hermes: ✓ 已记录 feedback 到 FEEDBACK.md，下次 self-improve 会采纳
```

---

## 六、跨渠道学习：把所有渠道学到的经验合并

Hermes 的多渠道部署让"用户偏好信号"从多条渠道来——Honcho 把这些全部合并到同一份 USER.md：

```
渠道              观察
---              -----
Telegram         用户偏好短句
WhatsApp         用户偏好更详细的回复（家庭情境下）
Slack（工作群）   用户偏好极简 + 任务驱动
CLI              用户偏好直接给完整代码
...
                    │
                    └─ Honcho 综合 ─► "技术内容要直接给完整代码；
                                     家庭对话可以加一些温度"
```

这是 OpenClaw 默认不做的——OpenClaw 把每个渠道视作独立 session，Hermes 把它们**综合**。

---

## 七、与 Reflective Agent / Reflexion 的关系

学术上"反思型 Agent"最有名的两个工作是：

- **Reflexion**（NeurIPS 2023）——让 Agent 用语言反思失败、写入长期记忆；
- **Self-Refine**（NeurIPS 2023）——类似思路，但每步迭代。

Hermes 的 Nudge Engine 在产品层做了 Reflexion / Self-Refine 的事——区别是：

| 维度 | Reflexion（学术） | Hermes Nudge |
|------|----------------|--------------|
| **数据源** | 单次任务 | 多次 + 跨渠道 + 跨 session |
| **触发** | 失败时 | 时间 + 事件 + 阈值 |
| **产物** | 反思文本（写入长期记忆） | Skill + USER.md 更新 |
| **可观测** | 不易观测 | `hermes nudge log` 全程可查 |
| **用户控制** | 无 | 细粒度开关 |

---

## 八、本节小结

| 主题 | 关键要点 |
|------|---------|
| Nudge 触发 | 时间 / 事件 / 阈值 / 外部 |
| Honcho Dialectic | 持续观察 + 提问 → 增量更新 USER.md |
| Weekly Review | 每周跑"Skill 健康度评估"，分类处理 |
| 跨渠道学习 | 把多渠道偏好信号合并到同一份 USER.md |
| 安全抑制 | 用户显式 feedback、confidence 阈值、冲突保留 |
| 与 Reflexion | Nudge = Reflexion 的产品化版本 |

---

*下一节：[15.7 借鉴哲学：从 Hermes 学到的"自进化"工程原则](./07_lessons_philosophy.md)*
