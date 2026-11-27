# 4.5 Practice: Personal Assistant Agent with Memory

Combining the knowledge from the previous four sections, we'll build a personal assistant that truly "remembers" — one that retains your preferences and continuously learns across sessions.

This project is a comprehensive application of everything in this chapter. We combine short-term memory (sliding window) with long-term memory (ChromaDB vector retrieval), plus automatic memory extraction, to build a personal assistant that genuinely "knows you".

### Core Design

The personal assistant's memory system operates in three coordinated layers:

1. **Short-term memory** (sliding window): maintains coherence within the current session, allowing the assistant to track conversation context
2. **Long-term memory** (ChromaDB): persists across sessions, remembering the user's identity, preferences, skills, and other long-term information
3. **Automatic extraction** (LLM analysis): after each conversation turn, automatically identifies information worth remembering

How these three layers collaborate: each time the user sends a message, the system first retrieves relevant information from long-term memory, then sends it along with recent conversation history to the LLM. After the LLM replies, the extractor analyzes whether the turn produced any new information worth storing. The entire process is transparent to the user.

## System Architecture

![Personal Assistant Agent Memory System Architecture](../svg/chapter_memory_05_memory_arch.svg)

## Complete Implementation

The `PersonalAssistant` class below is the core of the entire system. Pay attention to the processing flow in the `chat` method — this is exactly the three-layer memory working sequence: retrieve long-term memories → build messages (including short-term memory and relevant long-term memories) → call LLM → update short-term memory → auto-extract new memories.

Run the [complete example](https://github.com/Haozhe-Xing/agent_learning/blob/main/examples/chapter04/personal_assistant.py) from the repository root after setting `OPENAI_API_KEY`. Use the same username after restarting to retrieve stored memories. `ASSISTANT_MODEL` and `MEMORY_MODEL` select the chat and extraction models.

```bash
python -m pip install "chromadb>=1,<2" openai python-dotenv rich
python examples/chapter04/personal_assistant.py
```

```python
{{#include ../../../examples/chapter04/personal_assistant.py}}
```

## Key Implementation Details

Let's dive deeper into several important design decisions in the code:

**Relevance filtering in memory retrieval**: The `recall_memories` method sets a `relevance > 0.4` threshold. This is because vector retrieval always returns the "most similar" results, but those results aren't necessarily truly relevant. For example, if the user asks "what should I eat today", ChromaDB will still return some results even if there's no food-related information in the memory store (just with very low similarity). Setting a threshold filters out this noise.

**Trust boundary for retrieved memories**: Memories enter the user message as data. The system prompt describes how to use them. In production, validate memory provenance separately and give users a way to inspect and delete records.

**Extraction failures**: Extraction is synchronous in this example. If it fails, the CLI reports the failure and returns the reply already generated.

## Demo Output

```
[Alex]: My name is Alex, I'm a Python engineer working on an AI project
💾 Memory: User's name is Alex
💾 Memory: User is a Python engineer
💾 Memory: User is working on an AI project

Aria: Hi Alex! Great to meet you. As a Python engineer working on an AI project,
what direction are you focusing on? Agent development, model training, or something else?

[Alex]: Help me write a Python function to compute the Fibonacci sequence

Aria: (Based on the "Python engineer" memory, provides professional-grade code)
def fibonacci(n: int) -> list[int]:
    """Return the first n Fibonacci numbers"""
    ...

-- Next session --
[Alex]: Can that Fibonacci function from yesterday be optimized?

Aria: (Knows from memory that Alex is a Python engineer, gives targeted answer)
You can optimize it with a generator or memoization...
```

---

## Summary

This chapter built a complete memory system:

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Short-term memory | Sliding window | In-session coherence |
| Long-term memory | ChromaDB | Cross-session personalization |
| Memory extraction | LLM analysis | Auto-identify important information |
| Semantic retrieval | Vector similarity | Precise recall of relevant memories |

This framework can serve as the foundation for building personalized Agent applications.

---

_Next chapter: [4.6 Paper Readings: Frontiers in Agent Memory Systems](./06_paper_readings.md)_
