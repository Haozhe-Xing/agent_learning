# 4.7 Hands-On: MemGPT/Letta Memory Architecture Engineering Practice

> **Section Goal**: Based on the core ideas of MemGPT, implement a production-grade hierarchical memory Agent and understand how to use the Letta framework.

---

## From Paper to Engineering: Key Insights from MemGPT

Section 4.6 introduced the core ideas of the MemGPT paper — treating the LLM's context as analogous to an operating system's memory, using hierarchical storage and self-editing to break through context window limitations. This section turns those core ideas into runnable code.

MemGPT's engineered version **Letta** (renamed in 2025) provides a complete Agent memory management framework, but understanding the underlying principles is essential for custom development.

---

## Hierarchical Memory Architecture Implementation

### The Division of Labor in Three-Layer Memory

MemGPT draws an analogy between the LLM's context and an operating system's memory, adopting a "memory hierarchy" approach:

| Layer | What It Stores | OS Analogy | Key Characteristics |
|---|---|---|---|
| **Core Memory** | User name, long-term preferences, key facts, current goals | Resident memory / Registers | Always in the Prompt; serves as the model's "common knowledge base" |
| **Working Memory** | Recent context related to the current task | RAM | Limited capacity; keeps only the most recent entries |
| **Archive Memory** | Large volumes of historical content, long documents | Disk / External storage | Not in the Prompt; retrieved **on demand** |

**Why Layering Is Necessary**: The context window is a limited and expensive resource. If all history is stuffed into the Prompt, it quickly exceeds the window and becomes full of noise. The core idea of layering is — **keep the most important, most frequently used information resident (Core), temporarily buffer secondary information (Working), and place massive but infrequently used information externally for on-demand retrieval (Archive)**.

### Core Implementation

The key to the entire Agent lies in the `chat` method: auto-manage memory → build Prompt with memory → invoke the LLM → handle memory tool calls.

```python
class LayeredMemoryAgent:
    """A hierarchical memory Agent with three layers: Core + Working + Archive."""

    def __init__(self, model: str = "gpt-4.1"):
        self.model = model
        # Always in the Prompt: holds user profile and key preferences
        self.core_memory = {
            "user_name": "", "preferences": [],
            "key_facts": [], "active_goals": []
        }
        # Short-term information related to the current task
        self.working_memory = []
        # Persistent storage, simulating a vector database
        self.archive_memory = []
        # Conversation history
        self.conversation = []

    def chat(self, user_input: str) -> str:
        """Main conversation entrypoint: a 5-step pipeline."""
        # 1. Auto memory management: check if memory needs updating
        self._auto_manage_memory(user_input)
        # 2. Build the Prompt with memory
        messages = self._build_messages(user_input)
        # 3. Call the LLM
        response = client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=2000, tools=self._get_memory_tools()
        )
        # 4. Process tool calls (memory self-editing)
        reply = self._process_response(response)
        # 5. Save to conversation history
        self.conversation.append({"role": "user", "content": user_input})
        self.conversation.append({"role": "assistant", "content": reply})
        return reply

    def _build_messages(self, user_input: str) -> list[dict]:
        """Inject core memory + recent working memory into the System Prompt at all times."""
        system = f"""You are an Agent with hierarchical memory capabilities.
## Core Memory (Always Remember)
{json.dumps(self.core_memory, ensure_ascii=False, indent=2)}
## Working Memory (Current Task Related)
{json.dumps(self.working_memory[-5:], ensure_ascii=False, indent=2)}
## Memory Management Instructions
- Information in core memory is your "common knowledge" — always use it as the basis for your answers
- When asked about archived content, use the search_archive tool to retrieve it
- When you need to remember new information, use the update_core_memory tool
- When the current conversation produces content worth long-term storage, use the archive_content tool"""
        recent = self.conversation[-20:]  # Sliding window: last 10 turns
        return [{"role": "system", "content": system}] + recent \
             + [{"role": "user", "content": user_input}]

    # Remaining methods (_auto_manage_memory / _get_memory_tools /
    # _process_response / _search_archive) — see full implementation in the repo
```

> 📦 **Full code (~250 lines) is available in the repo** at `examples/chapter04/layered_memory_agent.py`, including the auto memory extraction prompt, schemas for 3 memory management tools, and tool call dispatch logic.

### Usage Example

```python
agent = LayeredMemoryAgent()

# Round 1: User introduces themselves
agent.chat("Hello! My name is Xiao Ming. I'm a data scientist, and I like using Python")

# Round 2: User expresses a preference
agent.chat("I prefer concise answers, not too wordy")

# Round 3: User talks about work
agent.chat("I'm working on a customer churn prediction project using XGBoost")

# Round 4: Verify that memory is retained
print(agent.chat("What project did I say I was working on?"))
# The Agent should recall "customer churn prediction project" from core memory
```

Across these four rounds, **core memory automatically captures** the user's name ("Xiao Ming"), identity ("data scientist"), preference ("concise answers"), and project ("customer churn prediction"). Even if the Agent restarts in the fifth round, as long as core memory is persisted, it can answer correctly.

---

## Quick Start with the Letta Framework

Letta (formerly MemGPT) is a commercial framework created by the paper's authors, providing complete hierarchical memory management:

```python
# pip install letta
from letta import create_client

letta_client = create_client()

# Create an Agent with hierarchical memory
agent = letta_client.create_agent(
    name="memory_assistant",
    memory_blocks=[
        {"label": "persona", "value": "You are a helpful AI assistant skilled at remembering user information."},
        {"label": "human",   "value": "User information to be filled in"},   # Agent auto-updates this
    ],
    llm="gpt-4.1",
    embedding="text-embedding-3-small",
)

# Chat with the Agent
response = letta_client.send_message(
    agent_id=agent.id,
    message="Hello! My name is Xiao Hong, and I'm doing NLP research",
    role="user"
)
# The Agent will automatically update "Xiao Hong" and "NLP research" into the human memory block
```

> 💡 **When to use Letta instead of building your own**:
> - Multi-user management, memory persistence, audit logs, model routing — Letta has already done these "non-core" capabilities
> - If "hierarchical memory" is your project's core innovation, building it yourself gives you more control
> - For simple prototypes or learning purposes, Letta is recommended (production-grade stability)

---

## Memory Decay and Forgetting Engineering

The human brain doesn't remember everything — important things are retained, unimportant things are gradually forgotten. An Agent's memory should work the same way:

```python
import math, time

# Memory types and their decay rates: identity never decays, trivial info decays rapidly
DECAY_RATES = {
    "identity": 0.0,    # Never decays
    "preference": 0.01, # Slow decay
    "fact": 0.05,       # Medium decay
    "context": 0.1,     # Fast decay
    "trivial": 0.3,     # Very fast decay
}

class MemoryWithDecay:
    """A memory system with decay + access reinforcement."""

    def __init__(self):
        self.memories: list[dict] = []  # Each entry has content/type/importance/created_at/access_count

    def add(self, content: str, type: str, importance: float = 0.5):
        self.memories.append({
            "content": content, "type": type, "importance": importance,
            "created_at": time.time(), "access_count": 0
        })

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Comprehensively considers: relevance, time decay, access reinforcement."""
        scored = []
        for mem in self.memories:
            relevance = self._compute_relevance(query, mem["content"])
            # Time decay: older memories have lower strength
            age_hours = (time.time() - mem["created_at"]) / 3600
            decay = math.exp(-DECAY_RATES.get(mem["type"], 0.05) * age_hours)
            # Access reinforcement: frequently retrieved memories are harder to forget
            access_bonus = min(0.2, mem["access_count"] * 0.02)
            # Composite score
            score = relevance * 0.4 + mem["importance"] * decay * 0.4 + access_bonus * 0.2
            scored.append((score, mem))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, mem in scored[:top_k]:
            mem["access_count"] += 1   # Increment access count (reinforce memory)
            results.append({
                "content": mem["content"], "score": score,
                "type": mem["type"], "age_hours": (time.time() - mem["created_at"]) / 3600
            })
        return results

    def cleanup(self, threshold: float = 0.01) -> str:
        """Clean up memories that have decayed below the threshold."""
        before = len(self.memories)
        self.memories = [m for m in self.memories if self._current_strength(m) > threshold]
        return f"Cleaned up {before - len(self.memories)} decayed memories, {len(self.memories)} remaining"
```

### Three Things Worth Discussing

1. **"Remember everything" is wrong**: Storage cost is a secondary concern — **retrieval quality** is the core issue. Trivial memories crowd out top_k slots, causing important information to be missed.
2. **Graded decay rates reflect the value difference of information**: Identity information is almost always useful, while trivial information quickly loses value — forgetting is not a flaw, but an active information filtering mechanism.
3. **Access reinforcement complements pure time-based decay**: Frequently retrieved memories receive a bonus ("the more you recall, the stronger it gets"), simulating the brain's memory consolidation.

---

## Summary

| Concept | Description |
|---|---|
| Hierarchical Memory | Three layers: Core + Working + Archive, corresponding to OS memory hierarchy |
| Self-Managed Memory | The Agent actively manages its own memory through tool calls (MemGPT's core idea) |
| Letta Framework | MemGPT's commercial version, providing complete hierarchical memory management |
| Memory Decay | Different memory types have different decay rates; important memories never decay |
| Access Reinforcement | Frequently retrieved memories are harder to forget (simulating the brain's "retrieval reinforcement") |

> 📖 **Further Reading**:
> - Packer et al. "MemGPT: Towards LLMs as Operating Systems." arXiv:2310.08560, 2023.
> - Letta Documentation. https://docs.letta.com, 2025.
> - Park et al. "Generative Agents: Interactive Simulacra of Human Behavior." UIST, 2023.

---

## 📝 Chapter Exercises

**Exercise 1 (Concept)**: MemGPT draws an analogy between the LLM's context and an operating system's memory. The hierarchical memory Agent implemented in this section has three layers: Core Memory, Working Memory, and Archive Memory. Please explain in your own words what each layer stores, why this layering is necessary, and what operating system concept each layer corresponds to.

<details>
<summary>Reference Answer</summary>

The division of labor across the three memory layers:

| Layer | What It Stores | OS Analogy | Key Characteristics |
|---|---|---|---|
| **Core Memory** | User name, long-term preferences, key facts, current goals | Resident memory / Registers | **Always** in the Prompt; serves as the model's "common knowledge base" |
| **Working Memory** | Recent context related to the current task | RAM | Limited capacity; keeps only the most recent entries |
| **Archive Memory** | Large volumes of historical content, long documents | Disk / External storage | Not in the Prompt; retrieved **on demand** |

**Why Layering Is Necessary:** The context window is a limited and expensive resource. If all history is stuffed into the Prompt, it quickly exceeds the window and becomes full of noise. The core idea of layering is — **keep the most important, most frequently used information resident (Core), temporarily buffer secondary information (Working), and place massive but infrequently used information externally for on-demand retrieval (Archive)**. This is exactly the operating system's "memory hierarchy (registers/RAM/disk)" approach: use limited fast storage for the hottest data, and place cold data in slow, high-capacity storage.

This both breaks through the physical limits of the context window and ensures that critical information (e.g., "the user is working on a customer churn prediction project") is never forgotten.

</details>

**Exercise 2 (Analysis)**: In this section's memory decay mechanism, the "identity" memory type has a decay rate of 0.0 (never decays), while the "trivial" type has a rate of 0.3 (very fast decay). A student says: "Since storage costs keep dropping, why not just never decay anything and remember everything — that seems like the safest approach." Is this idea correct? Please analyze it in terms of retrieval quality.

<details>
<summary>Reference Answer</summary>

This idea is **incorrect**. The problem is not storage cost, but **retrieval quality and context noise**.

- **"Remembering everything" drowns retrieval in noise**: The number of memories an Agent can place into context at each decision point is limited (top_k). If trivial information like "the user had a cup of coffee today" competes in ranking with critical information like "the user's name is Xiao Ming" or "the current project is customer churn prediction," trivial memories may crowd out valuable slots, causing truly important information to be missed.
- **The human brain works the same way**: At the beginning of this section, it's stated that "the human brain doesn't remember everything — important things are retained, unimportant things are gradually forgotten." Forgetting is not a flaw, but an **active information filtering mechanism**.
- **Graded decay rates reflect the value difference of information**: Identity information is almost always useful, so it doesn't decay; trivial information quickly loses value, so letting it decay rapidly and be cleaned up by `cleanup()` maintains the signal-to-noise ratio of the memory store.
- This section also designed **access reinforcement** (access_count): frequently retrieved memories receive a bonus and are harder to forget — this simulates the brain's "the more you recall, the stronger it gets" property, complementing pure time-based decay.

So the correct approach is **selective forgetting**: keep what's important, eliminate what's trivial, and reinforce what's repeatedly used. This way, what gets retrieved is truly relevant, high-quality memory.

</details>

**Exercise 3 (Hands-On)**: In this section's `MemoryWithDecay`, `_compute_relevance` uses simple "keyword overlap" to calculate relevance, which is very unfriendly to Chinese and synonyms. Please describe its shortcomings and write out the approach and core code for refactoring it to use **vector similarity (Embedding)**.

<details>
<summary>Reference Answer</summary>

**Shortcomings of keyword overlap:**
- Only matches **literally identical** words; cannot understand semantics. A query for "project" won't match "customer churn prediction work" because there are no shared words.
- Particularly unfriendly to Chinese: unlike English, Chinese doesn't use spaces for natural word segmentation, and `content.split()` cannot produce meaningful tokens.
- Cannot handle synonyms, hypernyms, or hyponyms (e.g., "phone" vs. "smartphone").

**Approach for refactoring with Embedding:** Encode both the query and memory content as vectors, and use **cosine similarity** to measure semantic closeness — semantically similar vectors are close in vector space, even when they share no words in common.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class MemoryWithDecay:
    def __init__(self):
        self.memories = []
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    def add(self, content, memory_type, importance=0.5):
        self.memories.append({
            "content": content, "type": memory_type, "importance": importance,
            "created_at": time.time(), "access_count": 0,
            "embedding": self.embedder.encode(content),   # Cache vector at ingestion time
        })

    def _compute_relevance(self, query: str, mem: dict) -> float:
        q_vec = self.embedder.encode(query)
        m_vec = mem["embedding"]
        cos = np.dot(q_vec, m_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(m_vec) + 1e-8)
        return (cos + 1) / 2   # Normalize to [0, 1]
```

**Explanation:**
- Encode and cache vectors at ingestion time (`add`); during retrieval, only the query needs to be encoded once, avoiding redundant computation.
- Cosine similarity measures the closeness of direction between two vectors, the standard approach for semantic retrieval; this way, "project" can match "customer churn prediction work."
- Normalizing to [0, 1] ensures it is on the same scale as `importance × decay` and `access_bonus` in `retrieve`, allowing proper weighted combination.
- This is essentially applying the idea from Section 4.3, "Long-Term Memory: Vector Databases and Retrieval," to the memory decay system — in real production, Archive Memory is typically implemented using a vector database (e.g., Milvus).

</details>

---

[4.6 Paper Reading: Advances in Memory Systems](./06_paper_readings.md)
