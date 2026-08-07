# 10.8 Agent-Specific Fine-Tuning: Teaching Models to Really Use Tools

> 🔧 *"General SFT teaches a model to talk; Agent SFT teaches it to act in the right format; Agentic-RL teaches it to actually get the job done under real feedback."*

Chapter 10.2 covered the general SFT + LoRA recipe. But when your goal is a model that **uses tools reliably, follows Agent behavior formats, and hallucinates fewer tool calls**, both data construction and training strategy need dedicated design.

This section focuses on one question: **how do you fine-tune a model that genuinely knows how to use tools?** Keep one boundary in mind throughout: Agent SFT is the starting point, not the finish line. It solves "can the model act in the right format"; Agentic-RL solves "once it acts, can it actually finish the task".

---

## SFT and Agentic-RL: Format First, Outcomes Second

Many teams treat SFT as the entire plan when training an Agent: collect a batch of expert trajectories, have the model imitate them, and expect it to complete tasks reliably in a real environment. But an Agent is not an ordinary chatbot — its output changes the outside world, so the training objective has to be split into two stages:

| Stage | Training objective | Capability learned | Main limitation |
|------|----------|------------|----------|
| **Agent SFT** | Imitate high-quality trajectories | Tool format, argument extraction, basic planning, answer style | No idea whether an action actually works once executed |
| **Agentic-RL** | Optimize the policy from environment feedback | Tool selection, error recovery, cost control, long-horizon task completion | Requires an executable environment and reward design |

In one sentence:

> **SFT teaches the model "what experts usually do"; Agentic-RL teaches the model "whether doing that actually works".**

So the Agent SFT in this section should be understood as the preparation stage for Agentic-RL: first give the model basic tool syntax and behavior format, then put it into an environment where it keeps learning from success, failure, and cost signals.

---

## The Limits of General SFT: Why Isn't It Enough?

A model trained on general dialogue data typically shows the following problems on Agent tasks:

> **❌ Hallucinated tool calls**: calling a tool that does not exist at all (e.g., emitting `search_google` when the actual tool list only contains `get_weather`, `web_search`, and `calculator`)
>
> **❌ Inconsistent format**: the same tool gets called differently every time (JSON format, function-call format, and natural language all mixed together)
>
> **❌ Broken multi-step reasoning**: step 3 forgets the result of step 1 (step 1 found the stock price = 150 yuan, yet step 3 says "I need to look up the stock price first")
>
> **❌ Tool abuse**: calling a tool even for trivial questions (the user asks "what is 3 + 5" and the model calls `calculator(expr="3+5")`)

The root cause of all of these: **tool-calling scenarios are rare and inconsistently formatted in general training data**. The fix is to re-fine-tune on purpose-built Agent trajectory data.

But even SFT on Agent trajectories still carries a fundamental limitation: it can only learn "what the expert output given this context", never "what happened in the environment after this action". A training sample can tell the model it should call `get_weather`, but it cannot let the model actually experience how to adjust its strategy when the API times out, the city name is ambiguous, permissions are insufficient, or the returned fields change.

This is precisely why Agentic-RL is needed afterwards: **tool calling is not a text-formatting problem, it is an environment-interaction problem**.

---

## Three Sources of Agent Fine-Tuning Data

### Source 1: Human Expert Annotation (Highest Quality, Highest Cost)

The most direct approach: have experts demonstrate correct Agent behavior trajectories.

```python
# A complete Agent SFT training sample (multi-turn trajectory format)
training_sample = {
    "id": "agent_001",
    "messages": [
        {
            "role": "system",
            "content": """You are an intelligent assistant with access to the following tools:

Tool list:
1. get_stock_price(ticker: str) -> float  # get a stock price
2. calculate(expr: str) -> float          # math calculation
3. search_news(query: str) -> list[str]   # search news

Rules:
- Only use tools from the list; never call a tool that does not exist
- Call exactly one tool at a time
- The tool-call format must be valid JSON"""
        },
        {
            "role": "user", 
            "content": "Look up Apple's (AAPL) recent stock price, and calculate how much 100 shares would cost."
        },
        {
            "role": "assistant",
            # Step 1: reasoning + tool call
            "content": "I need to look up the current AAPL price first, then compute the total cost.",
            "tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}]
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": "213.45"  # tool return value
        },
        {
            "role": "assistant",
            # Step 2: keep reasoning from the result
            "content": "AAPL currently trades at 213.45 USD; now computing the total cost of 100 shares.",
            "tool_calls": [{"name": "calculate", "arguments": {"expr": "213.45 * 100"}}]
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": "21345.0"
        },
        {
            "role": "assistant",
            # Step 3: combine the results into a final answer
            "content": "Apple (AAPL) currently trades at **$213.45**. Buying 100 shares would cost **$21,345** (excluding transaction fees)."
        }
    ]
}
```

**Key points**: every sample is a complete multi-turn trajectory containing:
1. A clear definition of the tool list
2. The reasoning at each step (the inner monologue)
3. Correctly formatted tool calls
4. Correct interpretation of tool return values
5. A final, integrated answer

### Source 2: Automatic Synthesis with a Strong Model (Most Efficient)

Use strong models such as GPT-4.1 / Claude Sonnet 4.5 to generate trajectory data in bulk, then filter it:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

SYNTHESIS_SYSTEM_PROMPT = """You are a data synthesis expert.
Given tool definitions and a user task, generate one correct Agent trajectory.

Requirements:
1. The reasoning must be clearly visible (explain why each step is taken)
2. Tool calls must strictly follow the JSON Schema
3. Handle tool errors correctly when they occur
4. Never call a tool outside the tool list
5. Answer simple questions directly; do not overuse tools

Output format: a complete conversation trajectory in JSON"""

async def synthesize_trajectory(
    tools: list[dict],
    task: str,
    model: str = "gpt-4.1"
) -> dict | None:
    """Synthesize one Agent trajectory with a strong model"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": f"""
Tool list: {tools}

User task: {task}

Please generate a complete Agent trajectory, including the reasoning and the tool calls.
"""}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # low temperature keeps the format consistent
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Synthesis failed for task '{task}': {e}")
        return None


async def batch_synthesize(
    tool_sets: list[list[dict]],
    task_pool: list[str],
    n_samples: int = 1000,
    concurrency: int = 20
) -> list[dict]:
    """Synthesize training data in bulk"""
    import random
    
    semaphore = asyncio.Semaphore(concurrency)  # concurrency control
    
    async def bounded_synthesize(tools, task):
        async with semaphore:
            return await synthesize_trajectory(tools, task)
    
    # Randomly pair tool sets with tasks
    pairs = [
        (random.choice(tool_sets), random.choice(task_pool))
        for _ in range(n_samples)
    ]
    
    results = await asyncio.gather(*[
        bounded_synthesize(tools, task) 
        for tools, task in pairs
    ])
    
    # Drop None values and malformed samples
    valid = [r for r in results if r is not None and validate_trajectory(r)]
    print(f"Synthesis success rate: {len(valid)}/{n_samples} = {len(valid)/n_samples:.1%}")
    return valid


def validate_trajectory(trajectory: dict) -> bool:
    """Validate that the trajectory format is legal"""
    try:
        messages = trajectory.get("messages", [])
        # Must contain system, user, and at least one assistant message
        roles = [m["role"] for m in messages]
        if "system" not in roles or "user" not in roles:
            return False
        if roles.count("assistant") < 1:
            return False
        # Check the tool-call format
        for msg in messages:
            if msg["role"] == "assistant" and "tool_calls" in msg:
                for call in msg["tool_calls"]:
                    if not all(k in call for k in ["name", "arguments"]):
                        return False
                    if not isinstance(call["arguments"], dict):
                        return False
        return True
    except (KeyError, TypeError):
        return False
```

### Source 3: Filtering Real User Interactions (Closest to Production)

Collect real user interactions from a live Agent system and use them as training data after quality filtering:

```python
class TrajectoryCollector:
    """Collect training trajectories from a live Agent"""
    
    def __init__(self, quality_threshold: float = 0.8):
        self.threshold = quality_threshold
    
    def collect_from_production(self, 
                                 raw_logs: list[dict]) -> list[dict]:
        """Filter high-quality trajectories out of production logs"""
        high_quality = []
        
        for log in raw_logs:
            score = self._quality_score(log)
            if score >= self.threshold:
                # Scrub PII and truncate over-long sequences
                cleaned = self._clean_trajectory(log)
                high_quality.append(cleaned)
        
        return high_quality
    
    def _quality_score(self, log: dict) -> float:
        """
        Quality scoring dimensions:
        - User satisfaction (thumbs up / rating / conversation continued)
        - Whether the task was completed (a final answer exists)
        - Whether tool calls succeeded (no error retries)
        - Whether the trajectory length is reasonable (no spinning in circles)
        """
        score = 0.0
        
        # Dimension 1: explicit user satisfaction
        if log.get("user_rating", 0) >= 4:
            score += 0.3
        elif log.get("conversation_continued"):  # user kept chatting = implicit satisfaction
            score += 0.15
        
        # Dimension 2: task completion
        messages = log.get("messages", [])
        last_msg = messages[-1] if messages else {}
        if last_msg.get("role") == "assistant" and len(last_msg.get("content", "")) > 50:
            score += 0.3
        
        # Dimension 3: tool-call success rate
        tool_calls = sum(1 for m in messages if m.get("role") == "tool")
        tool_errors = sum(1 for m in messages 
                         if m.get("role") == "tool" and "error" in str(m.get("content", "")).lower())
        if tool_calls > 0:
            success_rate = 1 - tool_errors / tool_calls
            score += 0.2 * success_rate
        
        # Dimension 4: trajectory efficiency (no idle spinning)
        n_turns = len([m for m in messages if m["role"] == "assistant"])
        if 1 <= n_turns <= 8:  # a reasonable trajectory length
            score += 0.2
        
        return score
    
    def _clean_trajectory(self, log: dict) -> dict:
        """Clean the trajectory: anonymize and normalize the format"""
        import re
        cleaned = log.copy()
        
        for msg in cleaned.get("messages", []):
            content = str(msg.get("content", ""))
            # Anonymize: phone numbers, emails, ID numbers
            content = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE]', content)
            content = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b', '[EMAIL]', content)
            msg["content"] = content
        
        return cleaned
```

---

## Training Data Formats Designed for Agents

Unlike general dialogue, Agent fine-tuning data has to cover many more special scenarios. The key is not to pile up as many "successful answers" as possible, but to let the model see three kinds of capability: when to act, how to act, and how to recover after an action fails.

### Scenario 1: Tool Selection Decisions (The Top Priority)

```python
# ✅ Correct: the right tool decision
{
    "input": "User: what is 2 + 2?",
    "output": "4",  # ← answer directly, no tool needed
    "annotation": "Simple arithmetic; answering directly is faster"
}

{
    "input": "User: what's the weather in Beijing today?",
    "tool_call": {"name": "get_weather", "args": {"city": "Beijing"}},
    "annotation": "Needs real-time information, so call the weather tool"
}

# ❌ Negative sample (include these too, for contrastive learning)
{
    "input": "User: what is your name?",
    "wrong_output": {"tool_call": "search_database(query='AI name')"},
    "correct_output": "I'm an AI assistant and I don't have a fixed name.",
    "error_type": "unnecessary_tool_use"
}
```

### Scenario 2: Error Recovery (Improving Robustness)

```python
# The right way to handle a failed tool call
error_recovery_sample = {
    "messages": [
        {"role": "user", "content": "Look up the TSLA stock price for me"},
        {"role": "assistant", 
         "content": "Looking up the TSLA price...",
         "tool_calls": [{"name": "get_stock_price", "arguments": {"ticker": "TSLA"}}]},
        {"role": "tool", 
         "content": "ERROR: API rate limit exceeded"},  # ← simulated failure
        {"role": "assistant", 
         # ✅ Correct: admit the failure and offer an alternative
         "content": "Sorry, the stock quote service is temporarily unavailable (API rate limit exceeded). "
                    "You can get real-time data by visiting Yahoo Finance directly or searching Google for 'TSLA stock price'."}
    ]
}
```

### Scenario 3: Multi-Tool Collaboration (Long-Chain Reasoning)

```python
# A complex task that requires several tool calls
multi_tool_sample = {
    "task": "Analyze how recent news affected the AAPL stock price",
    "trajectory": [
        # Step 1: search the news
        {"thought": "First get recent AAPL-related news"},
        {"tool_call": {"name": "search_news", "args": {"query": "AAPL Apple 2026"}}},
        {"tool_result": ["Apple Q2 earnings beat expectations...", "iPhone 18 series launched..."]},
        
        # Step 2: get the stock price
        {"thought": "Get the current price and historical prices for comparison"},
        {"tool_call": {"name": "get_stock_price", "args": {"ticker": "AAPL"}}},
        {"tool_result": "213.45"},
        
        # Step 3: analyze (no tool needed, pure reasoning)
        {"thought": "With both the news and the price in hand, I can do the analysis"},
        {"final_answer": "Based on an analysis of the recent news: ..."}
    ]
}
```

---

## Three Major Open-Source Agent Fine-Tuning Datasets

The community already offers ready-made, high-quality datasets you can fine-tune on directly:

### 1. Gorilla (Function Calling Specialist)

```python
# The Gorilla project: a dataset built specifically for API / function-calling training
# Source: UC Berkeley, with call examples for 1,600+ real APIs

from datasets import load_dataset

gorilla_data = load_dataset("gorilla-llm/APIBench", split="train")
# Format: user intent → API call → execution result

# Example sample
sample = {
    "instruction": "What are the symptoms of diabetes?",
    "api_call": 'requests.get("https://api.medlineplus.gov/v2/spellcheck", params={"terms": "diabetes"})',
    "provider": "medlineplus"
}
```

### 2. ToolLLM / ToolBench (Widest Coverage)

```python
# ToolBench: tool-calling data for 16,000+ real APIs
# Covers both single-tool and multi-tool scenarios, with full chains of thought

toolbench_data = load_dataset("ToolBench/ToolBench", split="train")
# Each sample averages 3-8 rounds of tool calls

# Structural highlights:
# - instruction: the user intent
# - tools: the list of available tools (varies per sample)  ← trains the model to adapt to different tool sets
# - conversations: the complete multi-turn trajectory (with CoT)
```

### 3. AgentInstruct (Microsoft, Highest Quality)

```python
# AgentInstruct (Microsoft, 2024):
# - 25M+ synthetic Agent trajectories
# - Covers code generation, RAG, multimodal, browser operation, and more
# - Used to train the Orca 3 / Phi-3 model families

# Key innovations:
# 1. Automatically generate complex variants from seed tasks (increasing difficulty diversity)
# 2. Score and filter generated trajectories with a reward model
# 3. Train per-domain specialists, then merge them

# Effect on Phi-3 Mini:
# 40%+ improvement over the base model on AgentBench
# Function-calling accuracy from 52% → 78%
```

---

## Training Configuration Designed for Agents

General SFT and Agent SFT differ in a few key training-configuration choices:

```python
from transformers import TrainingArguments
from trl import SFTTrainer

# Agent SFT-specific configuration
agent_training_args = TrainingArguments(
    output_dir="./agent-sft-output",
    
    # ① Batch size: Agent trajectories are usually longer, so shrink the batch size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # effective batch_size = 16
    
    # ② Learning rate: Agent tasks usually need a smaller learning rate
    learning_rate=5e-5,             # a bit lower than general SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    
    # ③ Sequence length: Agent trajectories are usually longer than dialogues
    max_seq_length=8192,            # make sure a full trajectory fits
    
    # ④ Epochs: Agent data is usually scarcer, so avoid overfitting
    num_train_epochs=2,             # 2-3 epochs is usually enough
    
    # ⑤ Compute the loss only on assistant replies (critical!)
    # Do not let the model "learn" the format of user input and tool output
)

# Key setting: response_template ensures only the assistant part is trained
trainer = SFTTrainer(
    model=model,
    args=agent_training_args,
    train_dataset=agent_dataset,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        # Compute the loss only on assistant tokens
        # loss mask = 0 for the user/system/tool parts
        label_pad_token_id=-100,
    ),
    formatting_func=format_trajectory_for_training,
)
```

```python
def format_trajectory_for_training(sample: dict) -> str:
    """
    Format an Agent trajectory into training text,
    and set the loss mask correctly (train only the assistant part)
    """
    messages = sample["messages"]
    
    # Use the ChatML format (supported by most models)
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        # Turn tool calls into a textual representation
        if "tool_calls" in msg:
            tool_call_str = json.dumps(msg["tool_calls"], ensure_ascii=False)
            content = f"{content}\n<tool_call>{tool_call_str}</tool_call>"
        
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    return formatted
```

---

## Evaluating an Agent-Specific Fine-Tuned Model

```python
class AgentEvaluator:
    """Agent-specific metrics for a fine-tuned Agent model"""
    
    def evaluate(self, model, test_cases: list[dict]) -> dict:
        results = {
            "tool_selection_accuracy": 0,   # tool selection accuracy
            "argument_accuracy": 0,          # argument-filling accuracy  
            "format_validity": 0,            # format validity rate
            "task_completion_rate": 0,       # task completion rate
            "unnecessary_tool_rate": 0,      # unnecessary tool-call rate
        }
        
        for case in test_cases:
            prediction = model.generate(case["input"])
            
            # 1. Tool selection: was the right tool name chosen?
            results["tool_selection_accuracy"] += (
                self._check_tool_selection(prediction, case["expected_tool"])
            )
            
            # 2. Argument accuracy: were the key arguments extracted correctly?
            results["argument_accuracy"] += (
                self._check_arguments(prediction, case["expected_args"])
            )
            
            # 3. Format validity: can it be parsed as valid JSON?
            try:
                json.loads(extract_tool_call(prediction))
                results["format_validity"] += 1
            except json.JSONDecodeError:
                pass
        
        n = len(test_cases)
        return {k: v/n for k, v in results.items()}
```

These metrics tell you whether the model "knows how to use tools", but not yet whether it "can complete the task". A model with very high tool-selection accuracy can still fail an 8-step task because it skipped checking one intermediate result. So Agent SFT evaluation should be split into two levels:

| Evaluation level | Typical metrics | Meaning |
|---------|---------|------|
| **Action-level evaluation** | Tool selection accuracy, argument accuracy, format validity | Has the model learned single-step tool calling? |
| **Trajectory-level evaluation** | Task completion rate, recovery success rate, average cost, human-takeover rate | Can the model actually finish multi-step tasks? |

If the action-level metrics are already high but the trajectory-level metrics remain unstable, the returns from piling on more SFT data are dropping — it is time to bring in environment feedback and trajectory-level rewards.

---

## When Should You Move from Agent SFT to Agentic-RL?

The goal of Agent SFT is not to train the model to "perfection", but to train it just enough to start exploring in an environment. Four signals tell you when to move on to Agentic-RL:

| Signal | Description | Next step |
|------|------|--------|
| **Format errors have dropped sharply** | Tool-call JSON is basically valid and argument fields are basically stable | You can start letting the model actually execute tools |
| **Tool selection is good enough to use** | Single-step tool selection accuracy has reached your business threshold | Introduce outcome rewards and tool success-rate rewards |
| **Long tasks still break down** | In multi-step tasks it forgets context, repeats calls, and cannot recover | Train planning and recovery with trajectory-level rewards |
| **The same failures keep recurring** | The same API errors, permission errors, and edge inputs fail again and again | Turn failed trajectories into preference pairs or RL rollouts |

Once you enter Agentic-RL, the training objective shifts from "predict the expert's next utterance" to "maximize the long-term return of the whole trajectory":

```text
SFT objective:
Given the context, predict the expert's next output.

Agentic-RL objective:
Given a task and an environment, maximize the final success rate, process reliability, and cost-effectiveness through a sequence of actions.
```

This is also why plain SFT is not enough to train a strong Agent: SFT sees static data, while Agentic-RL optimizes a dynamic closed loop.

---

## Practical Advice: Where to Start

```
Phase 1 (1-2 weeks): Data construction
├── Define your tool set (10-50 tools is a good starting point)
├── Synthesize 5,000-10,000 base trajectories with GPT-4.1
├── Filter out malformed and obviously wrong samples (keep 70-80%)
└── Manually spot-check 100 samples to assess the quality distribution

Phase 2 (1 week): Training
├── LoRA fine-tune on top of Llama 3.1 8B or Qwen2.5 7B
├── Train for 2 epochs, monitoring validation loss
└── At every 500-step checkpoint: quantitatively evaluate tool selection accuracy

Phase 3 (ongoing): Iteration
├── Collect real failure cases from production → add them to the training set
├── Fine-tune a new version every 2 weeks
└── A/B test: new model vs. old model on tool-call success rate
```

> 💡 **Rules of thumb**:  
> - 5,000 high-quality Agent trajectories > 50,000 low-quality general samples  
> - Data covering "failure recovery" is worth 2-3× as much as "successful trajectories"  
> - Randomizing the tool list (so each training run sees different tool combinations) dramatically improves the model's tool generalization

---

## Summary

| Dimension | General SFT | Agent SFT | Agentic-RL |
|------|---------|----------|------------|
| **Data format** | Single-turn dialogue | Multi-turn trajectories (with tool calls + results) | Rollouts in a real or simulated environment |
| **Key capability** | Language generation | Tool selection, argument extraction, error recovery | Long-horizon planning, environment adaptation, cost control |
| **Data volume** | 10K-1M | 1K-50K (quality first) | Depends on sampling cost and reward density |
| **Loss / objective** | Predict the reference answer | Imitate assistant tokens | Maximize trajectory-level reward |
| **Evaluation metrics** | BLEU/ROUGE | Tool-call accuracy, task completion rate | Success rate, recovery rate, stability, cost |
| **Core difficulty** | Language diversity | Format consistency + tool generalization | Reward design + exploration efficiency + safety boundaries |

> 🔗 **Relationship to Chapter 11 (Self-Evolving Agents)**: this section answers "how do I train my first usable Agent". [11.3 Agentic Data Flywheel](../chapter_self_evolving/03_data_flywheel.md) will show how to continuously turn real-world successes, failures, and environment feedback into training data, so the Agent gets stronger the more it is used.

---

## 📝 Exercises

After reading this chapter, close the book and answer the questions in your own words first, then expand the reference answers to check.

**Exercise 1 (Concept)**: This chapter repeatedly stresses that Agentic-RL follows a two-stage "SFT → RL" paradigm. Explain why both "SFT only" and "skip SFT and go straight to RL" are undesirable, and what irreplaceable role each stage plays. Also, the chapter mentions that "over-training with SFT actually hurts RL" — explain why.

<details>
<summary>Reference answer</summary>

One sentence captures the division of labor: **SFT teaches the model "what experts usually do"; RL teaches it "whether doing that actually works".**

**Why is SFT alone not enough?**
SFT is essentially "imitation learning" (like practicing calligraphy by tracing a master's strokes) — you show the model many expert demonstrations so it learns what to output for a given input. Its capability ceiling is nailed to the quality of the training data: a reasoning strategy that never appears in the data can never be learned. That is why pure SFT cannot make new abilities emerge — abilities like DeepSeek-R1's "self-reflection" and "long-chain reasoning" that were absent from the data.

**Why can't you skip SFT and go straight to RL?**
RL feels its way forward using reward signals. If you start from a base model, the model's output format is not even stable (unpaired tags, garbled tool-call syntax), so almost every sampled trajectory gets a reward of 0, the gradient signal is extremely weak, and training is wildly unstable and barely converges. SFT's job is to "initialize" the policy to a starting point with proper formatting and reasonable behavior, so RL has a decent basis for exploration.

**Why does over-training with SFT hurt RL?**
RL's progress comes from **exploration** — the model needs to generate diverse answers to the same question and learn from the contrast between good and bad ones. If SFT is pushed too hard, the model becomes extremely "confident" and outputs nearly the same answer for every input (its output distribution becomes too sharp), and diversity disappears. Then, in GRPO, the G answers sampled for the same question are nearly identical, the within-group standard deviation of rewards approaches 0, all advantages become zero after normalization, gradients vanish, and RL is effectively a no-op. That is why this chapter's "SFT graduation criteria" are: loss converged + format validity ≥ 90% + **output still diverse** — good enough is the goal, not as good as possible.

</details>

**Exercise 2 (Distinguish)**: A student says: "GRPO is just PPO with the Critic model deleted, so it must be weaker than PPO." Is that right? Explain, from the perspective of the **baseline**, the principle behind GRPO replacing the Critic with the within-group mean, and the price each approach pays. Then add: why does GRPO training "completely stall" if the sampling temperature is set too low (say 0.1)?

<details>
<summary>Reference answer</summary>

**The statement is wrong.** "Deleting the Critic" does not mean "getting weaker" — GRPO simply implements the Critic's core function in a more resource-efficient way.

**What does the Critic actually do?**
In PPO, the Critic has exactly one essential job: provide a **baseline** that converts "absolute reward" into "relative advantage", thereby reducing gradient variance. An analogy: scoring 85 on a test means nothing by itself — you have to compare it with the class average. If the average is 60 you did great (positive advantage); if the average is 90 you underperformed (negative advantage). The Critic is the model that "predicts the class average".

**How does GRPO replace it?**
GRPO's insight: if all we need is a baseline, then just **sample G answers for the same question and use the mean reward of those G answers as the baseline**:

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

Answers better than the group average get a positive advantage (reinforced); worse ones get a negative advantage (suppressed). This removes the need to train and store a Critic the same size as the main model, cutting GPU memory from roughly 3× model size down to about 1.5×.

**The price each pays:**
- The Critic is a "parameterized function approximator". In theory it generalizes to unseen states, but it needs extra training, carries estimation error, and that error propagates into policy updates, adding instability.
- The within-group mean is a "non-parametric statistic". It needs no training and has no error propagation, but it **depends on sampling quality** — you need G extra samples per question (raising sampling cost), and if sampling diversity is insufficient, the baseline is inaccurate.

**Why does a too-low temperature stall training?**
All of GRPO's learning signal comes from "the reward differences among answers within a group". Temperature controls sampling randomness/diversity: if the temperature is too low (e.g., 0.1), the G answers generated for the same question are nearly identical → their rewards are nearly identical → the within-group standard deviation $\sigma_r \approx 0$ → all normalized advantages ≈ 0 → gradients are zero → parameters stop updating and training completely stalls. That is why this chapter recommends a temperature of 0.6–0.8: enough variation among answers to compare, without being so random that quality collapses.

</details>

**Exercise 3 (Hands-on)**: Suppose you want to train a "math-solving Agent" that must first emit `<think>...</think>` with its reasoning and then give the final numeric answer. Write a Python reward function `math_reward(completion, ground_truth)` that combines at least three dimensions: (1) whether the answer is correct; (2) whether the format is well-formed (a non-empty think block exists); (3) defense against the reward hack of "writing gibberish inside think and just producing the correct answer". Finally, explain: if this reward function is used in GRPO training and you sample 8 answers for the same problem, how are the within-group normalized advantages computed?

<details>
<summary>Reference answer</summary>

Core idea: **do not look only at whether the final answer is right**, otherwise the model learns the shortcut of "writing gibberish in think and pulling out the answer". You have to tie "answer correctness × the authenticity of the reasoning" together.

```python
import re

def math_reward(completion: str, ground_truth: str) -> float:
    """
    Composite reward for a math-solving Agent, return value ∈ [0, 1]
    Three dimensions: answer correctness + format compliance + anti-reward-hacking
    """
    reward = 0.0

    # ── Dimension 1: format compliance (think block exists and is non-empty) ──
    has_think = "<think>" in completion and "</think>" in completion
    think_content = ""
    if has_think:
        think_content = completion.split("<think>")[1].split("</think>")[0].strip()
        if len(think_content) >= 20:   # substantive reasoning, not an empty shell
            reward += 0.3

    # ── Dimension 2: answer correctness ───────────────────────────────────
    answer_correct = False
    try:
        # Take the last number in the completion as the final answer
        nums = re.findall(r'-?[\d,]+\.?\d*', completion)
        pred = float(nums[-1].replace(",", ""))
        true = float(ground_truth.replace(",", ""))
        rel_err = abs(pred - true) / (abs(true) + 1e-8)
        answer_correct = rel_err < 1e-2     # allow 1% relative error
    except (ValueError, IndexError):
        answer_correct = False

    if answer_correct:
        reward += 0.5

    # ── Dimension 3: anti-reward-hacking (think must be "normal language") ──
    # Measure the fraction of valid characters (CJK/Latin letters, digits, common
    # punctuation) in think; a very low fraction suggests gibberish used to fake
    # an answer → discount the reward even if the answer is correct
    if think_content:
        valid = len(re.findall(r'[\u4e00-\u9fff\w\s.,!?，。！？；：=+\-*/()]', think_content))
        coherence = valid / max(len(think_content), 1)
        if coherence >= 0.8:
            reward += 0.2          # coherent reasoning, full marks
        elif coherence < 0.5 and answer_correct:
            reward *= 0.5          # right answer but gibberish reasoning → halve the reward, closing the hack

    return max(0.0, min(1.0, reward))
```

**Key design points:**
- A correct answer earns only 0.5, leaving room for "format + genuine reasoning" to count as well, so the model does not fixate on the answer alone.
- Dimension 3 is the core defense: when think looks like gibberish (valid-character ratio < 0.5) but the answer happens to be right, it is judged as "faking the answer" and the total reward is halved — making the loophole unprofitable.

**How are the within-group normalized advantages computed under GRPO?**
For the same problem, sample 8 answers $y_1,\dots,y_8$ with the old policy and compute their rewards $r_1,\dots,r_8$. Then normalize within the group:

```python
import numpy as np

def grpo_advantages(rewards, eps=1e-8):
    r = np.array(rewards, dtype=np.float64)
    mu, sigma = r.mean(), r.std()
    if sigma < eps:          # all 8 rewards are identical → good and bad are indistinguishable
        return [0.0] * len(r)
    return ((r - mu) / (sigma + eps)).tolist()

# Example: rewards of 8 answers
rewards = [0.93, 0.30, 1.00, 0.50, 0.30, 1.00, 0.45, 0.30]
print(grpo_advantages(rewards))
# Answers above the mean → positive advantage (reinforce their reasoning paths)
# Answers below the mean → negative advantage (suppress their reasoning paths)
```

In other words: first compute the mean $\mu_r$ and standard deviation $\sigma_r$ of these 8 rewards, then use $\hat{A}_i=(r_i-\mu_r)/(\sigma_r+\epsilon)$ to turn "absolute reward" into "how good or bad relative to this group's average". This advantage multiplies the policy gradient of every token in that answer, so "the whole reasoning path of a good answer is reinforced, and that of a bad answer is suppressed". Note one edge case: if all 8 rewards are identical ($\sigma_r\approx0$), all advantages are 0 and nothing updates — which is exactly the training-stall phenomenon caused by too low a temperature in Exercise 2.

</details>

---

## References

1. Patil et al. "Gorilla: Large Language Model Connected with Massive APIs." NeurIPS 2023.
2. Qin et al. "ToolLLM: Facilitating LLMs to Master 16000+ Real-world APIs." ICLR 2024.
3. Mitra et al. "AgentInstruct: Toward Generative Teaching with Agentic Flows." Microsoft Research 2024.
4. Liu et al. "What Makes Good Data for Alignment? (DEITA)" ICLR 2024.
5. Wang et al. "Self-Instruct: Aligning Language Models with Self-Generated Instructions." ACL 2023.
