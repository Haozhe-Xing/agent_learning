# 15.6 Nudge Engine & Cross-Session Learning

> ☤ *"The smartest Agent isn't the fastest reactor — it's the one that proactively asks 'what did I learn?'"*

---

## What the Nudge Engine Is

Self-Evolving (15.4) is *passive* — it evaluates only after each task. The Nudge Engine is *proactive*: every so often, Hermes asks itself:

- What did I learn recently worth persisting?
- Has my model of the user changed?
- Can I distill the past days' experience into a Skill?

---

## 4 Trigger Mechanisms

| Trigger | Example |
|---------|---------|
| Time | every 30 min → "worth distilling?"; every 24h → "update user model?"; weekly → "skill review" |
| Event | session ends → distill; ≥3 tool failures → update skill/USER.md; "remember X" → write MEMORY.md |
| Threshold | >100 messages → compress; weekend → weekly review |
| External | cron task → run a specific skill |

```python
class NudgeLoop:
    async def maybe_nudge(self, trigger):
        if not self._enabled(trigger): return
        if trigger == "task_done":
            await self._maybe_extract_skill()
        elif trigger == "user_pref_changed":
            await self._update_user_model()
        elif trigger == "weekly_review":
            await self._weekly_skill_review()
```

---

## Honcho Dialectic: User Modeling

Honcho continuously observes conversational signals and distills them into `USER.md`, updated **incrementally** (never rewritten wholesale):

```json
{
  "communication_style": "terse, structured, no emoji",
  "working_hours": "deep work morning, meetings afternoon",
  "tools_primary": ["Python", "Node"],
  "values": ["correctness", "deep detail"],
  "avoids": ["emoji-heavy text", "marketing speak"]
}
```

Suppression rules prevent runaway self-speculation: skip when no new signal, confidence < 0.6, conflicts with existing content, or explicit "forget that".

---

## Weekly Skill Review

Once a week, Hermes scores every skill on invocation count, success rate, token cost, and explicit feedback — classifying each as healthy / needs-improvement / dying, then writing a summary to `MEMORY.md`.

## Cross-Channel Learning

Honcho merges preference signals from all channels (Telegram short / WhatsApp detailed / Slack minimal / CLI code) into one `USER.md` — a synthesis OpenClaw doesn't do by default.

## vs Reflexion (the Academic Predecessor)

| Dimension | Reflexion (academic) | Hermes Nudge |
|-----------|---------------------|--------------|
| Data source | single task | multi + cross-channel + cross-session |
| Trigger | on failure | time + event + threshold |
| Product | reflection text | Skill + USER.md update |
| User control | none | fine-grained switches |

> Nudge is Reflexion, productized.

---

## Section Summary

| Topic | Key point |
|-------|-----------|
| Triggers | time / event / threshold / external |
| Honcho | observe → incremental USER.md update |
| Weekly review | skill health scoring |
| Cross-channel | merge signals into one USER.md |
| Safety | feedback, confidence threshold, conflict-keep |

---

*Next section: [15.7 Borrowing the Philosophy](./07_lessons_philosophy.md)*
