# 10.5 GRPO/GSPO: Group Relative Policy Optimization and Reward Function Design

In [10.3 PPO: Proximal Policy Optimization](./03_ppo.md) and [10.4 DPO: Direct Preference Optimization](./04_dpo.md), we introduced the PPO and DPO policy optimization algorithms respectively. PPO requires an additional Critic model (large memory footprint), while DPO is simple but completely offline (it cannot explore new strategies).

**GRPO (Group Relative Policy Optimization)** [1] is an algorithm tailor-made by the DeepSeek team for large-model RL training. It replaces the Critic model with **within-group sampling comparison**, dramatically reducing resource consumption while retaining online exploration capability. **GSPO (Group Sequence Policy Optimization)** [10] is Alibaba's Qwen team's improvement on top of GRPO: by raising the optimization granularity from the token level to the sequence level, it solves the training stability problems of large-scale models (especially MoE architectures). This section also introduces the core driving force behind both algorithms — **reward function design** — because the reward function defines "what counts as good Agent behavior" and therefore directly determines training effectiveness.

---

### 3.1 GRPO's Core Insight

GRPO [1] is an algorithm tailor-made by the DeepSeek team for large-model RL training. Its core insight is:

> **PPO's Critic model is essentially just providing a "baseline" to reduce the variance of advantage estimation. For language models, there is a simpler way to obtain a baseline — sample multiple responses to the same question and use the within-group mean as the baseline.**

This insight brings enormous practical value:

| Dimension | PPO | GRPO | Improvement |
|-----------|-----|------|-------------|
| **Number of models** | Policy + Critic + Reference | Policy + Reference | **One fewer Critic** |
| **Memory requirement** | ≈ 3× model size | ≈ 1.5× model size | **~50% savings** |
| **Training stability** | Critic errors propagate to the Policy | No Critic error propagation | **More stable** |
| **Hyperparameters** | Many (GAE λ, Critic lr, ...) | Few (clip ε, KL β, G) | **Easier to tune** |

### 3.2 Within-Group Sampling and Normalization: Replacing the Critic with "Group Comparison"

GRPO's core operation is as follows:

For each input $x$, sample $G$ responses using the current policy (its old version $\pi_{\theta_{old}}$):

$$\{y_1, y_2, \ldots, y_G\} \sim \pi_{\theta_{old}}(\cdot | x)$$

Then compute the reward $r_i = R(x, y_i)$ for each response and perform **within-group normalization**:

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$$

Where:

$$\mu_r = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_r)^2}$$

Term-by-term interpretation:

- $\mu_r$: **within-group reward mean** — the average reward of the $G$ responses to the same question, acting as the "baseline" (the Critic's substitute)
- $\sigma_r$: **within-group reward standard deviation** — used for normalization, removing the influence of the absolute reward scale
- $\epsilon$: numerical stability constant (typically $10^{-8}$), prevents division by zero
- $\hat{A}_i > 0$: the $i$-th response is better than the within-group average → it should be **reinforced**
- $\hat{A}_i < 0$: the $i$-th response is worse than the within-group average → it should be **suppressed**

**Statistical properties of normalization**:

1. **Zero mean**: $\sum_i \hat{A}_i \approx 0$ — half of the responses are reinforced, half are suppressed (relative comparison)
2. **Unit variance**: $\text{Var}(\hat{A}_i) \approx 1$ — gradient magnitude is unaffected by the reward scale

**Why can the within-group mean replace the Critic?** The core argument:
- The Critic's role = provide a baseline → convert absolute rewards into relative advantages → reduce gradient variance
- The within-group mean likewise provides a baseline → likewise converts absolute rewards into relative advantages → likewise reduces gradient variance
- **Difference**: the Critic is a parameterized function approximator (it needs training and may carry estimation error); the within-group mean is a non-parametric statistic (no training needed, but it depends on sampling quality)
- **Cost**: GRPO must sample $G$ responses per question (higher sampling cost), whereas PPO only needs 1

```python
import numpy as np

def compute_grpo_advantages(rewards: list[float], eps: float = 1e-8) -> list[float]:
    """
    Compute GRPO within-group normalized advantages
    
    Args:
        rewards: reward values of the G responses to the same question [r₁, r₂, ..., r_G]
        eps: numerical stability constant
    
    Returns:
        list of normalized advantage values [Â₁, Â₂, ..., Â_G]
    
    Properties:
        - Σ Â_i ≈ 0 (zero mean)
        - Var(Â_i) ≈ 1 (unit variance)
    """
    rewards = np.array(rewards, dtype=np.float64)
    mu = rewards.mean()
    sigma = rewards.std()
    
    if sigma < eps:
        # All responses have the same reward → good and bad cannot be told apart → advantage is zero
        return [0.0] * len(rewards)
    
    advantages = (rewards - mu) / (sigma + eps)
    return advantages.tolist()


# ── Example ──────────────────────────────────────────────────────────
# Same math problem, the model generates 8 responses: 5 correct, 3 wrong
rewards = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
advantages = compute_grpo_advantages(rewards)

print("Rewards:    ", rewards)
print("Advantages: ", [f"{a:+.3f}" for a in advantages])
# Correct answers (r=1.0) → advantage ≈ +0.667 → reinforce these reasoning paths
# Wrong answers (r=0.0)   → advantage ≈ -1.333 → suppress these reasoning paths
# Note: |negative advantage| > |positive advantage|; wrong answers are suppressed more strongly
```

### 3.3 GRPO's Complete Objective Function

GRPO's optimization objective combines PPO's Clip mechanism with a KL divergence constraint:

$$\mathcal{L}_{GRPO}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[ \min\left( \rho_{i,t} \hat{A}_i,\ \text{clip}\left(\rho_{i,t}, 1-\epsilon, 1+\epsilon \right) \hat{A}_i \right) - \beta \cdot \mathbb{D}_{KL}\left[\pi_\theta \| \pi_{ref}\right] \right]$$

Term-by-term interpretation:

- $\frac{1}{G} \sum_{i=1}^{G}$: average over the $G$ responses — every response contributes equally to the gradient
- $\frac{1}{|y_i|} \sum_{t=1}^{|y_i|}$: average over the tokens of the $i$-th response — prevents long responses from dominating the gradient (length normalization)
- $\rho_{i,t} = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$: the importance sampling ratio of the $t$-th token of the $i$-th response
- $\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, ...) \hat{A}_i)$: the PPO Clip policy loss — inherited from PPO, prevents any single update step from being too large
- $\beta \cdot \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$: the KL divergence penalty — prevents the policy from drifting too far from the SFT model, avoiding reward hacking and language degeneration. For a detailed explanation of KL divergence, see [Appendix E: KL Divergence Explained](../appendix/kl_divergence.md)

### 3.4 GRPO Training Architecture and Workflow

![GRPO Training Architecture](../svg/chapter_agentic_rl_03_grpo_architecture.svg)

![GRPO Single Training Iteration Workflow](../svg/chapter_agentic_rl_03_grpo_iteration.svg)

> 🎬 **Interactive animation**: Get hands-on with the core GRPO process — G=8 within-group sampling, reward scoring, normalized advantage computation, and probability distribution updates — and build an intuition for the elegant idea of "replacing the Critic with group comparison."
>
> <a href="../animations/grpo_sampling.html" target="_blank" style="display:inline-block;padding:8px 16px;background:#E91E63;color:white;border-radius:6px;text-decoration:none;font-weight:bold;">▶ Open the GRPO Within-Group Sampling Interactive Animation</a>

### 3.5 Complete GRPO Implementation with TRL

```python
"""
Complete implementation of GRPO training
Based on the GRPOTrainer from the Hugging Face TRL library
"""

from trl import GRPOConfig, GRPOTrainer

# ── GRPO training configuration ───────────────────────────────────────────
grpo_config = GRPOConfig(
    output_dir="./checkpoints/grpo",

    # GRPO core parameters
    num_generations=8,               # G=8: balances advantage estimation quality against sampling cost
                                     # G too small → high variance; G too large → high compute cost

    # Training hyperparameters
    num_train_epochs=2,
    per_device_train_batch_size=1,   # G responses must be generated, so keep the batch size small
    gradient_accumulation_steps=8,   # Effective batch size = 1 × 8 = 8
    learning_rate=5e-6,              # RL-phase learning rate ≈ 1/40 of the SFT learning rate
                                     # Too large collapses the policy; too small converges extremely slowly
    warmup_ratio=0.1,
    max_grad_norm=0.5,               # Gradient clipping, prevents gradient explosion during RL training

    # Generation parameters
    max_new_tokens=512,
    temperature=0.7,                 # Guarantees diversity among the G responses
                                     # Temperature too low → responses converge → all advantages become 0

    # GRPO algorithm parameters
    kl_coef=0.01,                    # β: KL divergence penalty coefficient
                                     # Too large → the policy cannot optimize enough; too small → the policy drifts too far

    # Precision and performance
    bf16=True,

    # Logging and checkpoints
    logging_steps=1,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=3,
    report_to="tensorboard",
)


# ── Reward function definition ────────────────────────────────────────────
def reward_function(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """
    Comprehensive reward function for Agent behavior quality (example implementation)

    For detailed reward function design methods, see the "Reward Function Design" section below
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # Dimension 1: format correctness
        has_think = "<think>" in completion and "</think>" in completion
        if has_think:
            reward += 0.2
            think_content = completion.split("<think>")[1].split("</think>")[0].strip()
            if len(think_content) > 20:
                reward += 0.1   # Contains substantive reasoning

        # Dimension 2: soundness of the tool call
        if "<tool_call>" in completion and "</tool_call>" in completion:
            reward += 0.3
            try:
                tool_str = completion.split("<tool_call>")[1].split("</tool_call>")[0].strip()
                if "(" in tool_str and ")" in tool_str:
                    reward += 0.2   # Function call syntax is correct
            except IndexError:
                reward -= 0.1   # Tags are not paired

        # Dimension 3: efficiency penalty
        num_tool_calls = completion.count("<tool_call>")
        if num_tool_calls > 5:
            reward -= 0.1 * (num_tool_calls - 5)

        rewards.append(max(0.0, reward))

    return rewards


# ── Initialize and launch training ────────────────────────────────────────
trainer = GRPOTrainer(
    model=model,                     # Model trained during the SFT phase
    config=grpo_config,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    reward_funcs=reward_function,
)

print("🚀 Starting GRPO training...")
trainer.train()
trainer.save_model("./checkpoints/grpo-final")
print("✅ GRPO training complete!")
```

---

## Systematic Comparison of the Three Algorithms

### 4.1 Architecture Comparison

| Dimension | PPO | DPO | GRPO |
|-----------|-----|-----|------|
| **Models required** | Policy + Critic + Reference | Policy + Reference | Policy + Reference |
| **Memory requirement** | ≈ 3× model size | ≈ 2× model size | ≈ 1.5× model size |
| **Training data** | Online sampling + reward model | Offline preference pairs | Online sampling + reward function |
| **Advantage estimation** | GAE (depends on the Critic) | None (implicit reward margin) | Within-group normalization (no Critic) |
| **Update constraint** | Clip + KL | Implicit KL (via $\beta$) | Clip + KL |

### 4.2 Training Characteristics Comparison

| Dimension | PPO | DPO | GRPO |
|-----------|-----|-----|------|
| **Training stability** | Medium (Critic error propagation) | High (supervised learning) | High (no Critic error) |
| **Number of hyperparameters** | Many (≥6) | Very few ($\beta$ only) | Few (≤4) |
| **Data efficiency** | Low (needs online sampling) | High (offline reuse) | Medium (needs G× sampling) |
| **Explorability** | Strong (online RL) | None (purely offline) | Strong (online RL) |
| **Capability ceiling** | Can exceed the data | Limited by preference data quality | Can exceed the data |

### 4.3 Algorithm Selection Guide

![RL Algorithm Selection Decision Tree](../svg/chapter_agentic_rl_05_algo_decision_tree.svg)

### 4.4 Empirical Performance

| Project | Algorithm | Core Achievement |
|---------|-----------|-----------------|
| **InstructGPT** [2] | PPO | Proved that RLHF can greatly improve instruction following |
| **Llama 2** [3] | PPO | Safety alignment for a 70B model |
| **Zephyr** [4] | DPO | A 7B model surpasses the PPO baseline using DPO |
| **DeepSeek-R1** [5] | GRPO | Emergent long-chain reasoning; math/code ability rivals o1 |
| **DeepSWE** [6] | GRPO | SWE-bench Verified 59% (open-source SOTA) |

---

## Key Monitoring Metrics and Tuning Guide

During RL training (PPO or GRPO), the following metrics are the core basis for judging whether training is healthy:

| Metric | Healthy Range | Warning Signal | Remedy |
|--------|--------------|----------------|--------|
| `mean_reward` | Should rise steadily | Flat for a long time or falling | Check the reward function design, lower the KL coefficient |
| `kl_divergence` | < 10–15 nats | Keeps growing | Increase the KL coefficient $\beta$ |
| `clip_fraction` | 0.1–0.3 | > 0.5 | Lower the learning rate or increase clip $\epsilon$ |
| `mean_ratio` | Close to 1.0 | Keeps drifting away from 1.0 | Lower the learning rate, add more warmup |
| `reward_std` | > 0 (within-group variation) | ≈ 0 | Raise the temperature, check the reward function |

> **📌 Engineering Practice Notes**
>
> - **Choosing the group size $G$** (GRPO): $G = 4$–$16$ is the common range. Too small a $G$ gives high-variance advantage estimates; too large a $G$ makes sampling expensive. Start with $G = 8$.
> - **Temperature** (GRPO): 0.6–0.8 is recommended. If the temperature is too low, the $G$ responses may be completely identical, making $\sigma_r \approx 0$ and all advantages zero.
> - **Learning rate**: the RL-phase learning rate is typically $\frac{1}{10}$ to $\frac{1}{50}$ of the SFT phase. Too large a learning rate collapses the policy within a few steps.
> - **Gradient clipping**: `max_grad_norm=0.5` is recommended; gradient explosion is more common in RL training than in SFT.
> - **Tuning $\beta$** (DPO): $\beta$ is usually 0.1–0.5. Too small a $\beta$ → unstable training; too large a $\beta$ → the policy barely updates.

---

## Reward Function Design — Formalizing Goals as Optimizable Signals

### 5.1 The Central Role of the Reward Function

In the GRPO training framework, **the reward function $R: \mathcal{X} \times \mathcal{Y} \to \mathbb{R}$ is the only bridge connecting "human intent" and "model behavior"**. It turns our intuitive judgment about what makes a "good Agent" into a differentiable (or samplable) numerical signal, and therefore directly determines the direction of reinforcement learning optimization.

The core challenge of reward function design is:

$$\text{True objective} \neq \text{Computable proxy metric}$$

**Why are the two not equivalent?** The true objective is usually a vague subjective judgment (e.g., "high output quality," "high user satisfaction"), whereas a computable proxy metric must be a concrete number (e.g., "test case pass rate," "format compliance rate"). This gap is the fundamental source of **reward hacking** [7] — the model finds shortcuts that maximize the proxy metric, and those shortcuts often violate the true intent.

**A typical case**: if the reward function only checks whether the final answer is correct, the model may learn to emit gibberish inside `<think>` and then produce the right answer anyway — a high reward, but completely meaningless reasoning. This is the classic gap between the proxy metric (answer correctness) and the true objective (meaningful reasoning).

#### Four Basic Principles of Reward Function Design

| Principle | Formal Description | Consequence of Violation |
|-----------|-------------------|--------------------------|
| **Verifiability** | The reward is based on objectively computable criteria rather than subjective judgment | Noisy reward signal, unstable training |
| **Multi-dimensional coverage** | $R = \sum_k w_k R_k$, covering multiple quality dimensions of the task | The model over-optimizes one dimension and ignores the rest |
| **Density** | Provide reward signals at multiple timesteps of the trajectory instead of only at termination | Sparse rewards make credit assignment hard and slow down convergence |
| **Robustness** | The reward function resists the model's "gaming" behavior | The model learns reward hacking: high reward, low real quality |

**Reading the multi-dimensional combination formula $R = \sum_k w_k R_k$**: each dimension's reward $R_k \in [0, 1]$ is computed independently, and the weights $w_k$ satisfy $\sum_k w_k = 1$. The choice of weights expresses the relative importance of each dimension: accuracy gets the highest weight (it is the core of the task), safety the lowest (it is rarely triggered in most cases).

### 5.2 Design and Implementation of Core Reward Dimensions

#### Dimension 1: Accuracy Reward

The accuracy reward is the most central dimension: it directly measures whether the Agent completed the task correctly. Different task types need different evaluation methods:

```python
import re
from typing import Optional

def accuracy_reward(
    prediction: str,
    ground_truth: str,
    task_type: str = "math",
    tolerance: float = 1e-2,
) -> float:
    """
    Accuracy reward: evaluate whether the Agent's output completes the task correctly
    
    Args:
        prediction:   the model's full output (including the reasoning process)
        ground_truth: the reference answer
        task_type:    task type, determines the evaluation method
        tolerance:    relative error tolerance for numerical comparison
    
    Returns:
        reward value ∈ [0, 1]
    """
    if task_type == "math":
        # Math task: extract the final number from the output, allow a relative error of `tolerance`
        try:
            pred_num = _extract_final_number(prediction)
            true_num = float(ground_truth.replace(",", ""))
            relative_error = abs(pred_num - true_num) / (abs(true_num) + 1e-8)
            return 1.0 if relative_error < tolerance else 0.0
        except (ValueError, AttributeError):
            return 0.0

    elif task_type == "code":
        # Code task: run the test cases and score by pass rate (partial reward)
        # 
        # Why use a partial reward instead of a 0/1 reward?
        # A 0/1 reward (sparse reward) makes credit assignment hard:
        #   - If the model passes 9/10 test cases, a 0/1 reward gives 0 and cannot tell "nearly correct" from "completely wrong"
        #   - A partial reward k/n provides a denser gradient signal, helping the model improve step by step
        # This matches the idea of Curriculum Learning: first learn to pass the easy tests, then tackle the hard ones
        code = _extract_code_block(prediction)
        if not code:
            return 0.0
        test_results = _run_test_cases(code, ground_truth)
        # Partial reward: passing k/n test cases earns k/n points
        return test_results["passed"] / max(test_results["total"], 1)

    elif task_type == "tool_call":
        # Tool-call task: check whether the tool name and arguments are correct
        pred_call = _parse_tool_call(prediction)
        true_call = _parse_tool_call(ground_truth)
        if pred_call is None:
            return 0.0
        score = 0.0
        if pred_call.get("name") == true_call.get("name"):
            score += 0.5   # Tool name is correct
        if pred_call.get("args") == true_call.get("args"):
            score += 0.5   # Arguments match exactly
        return score

    else:
        # Generic: exact string match
        return 1.0 if prediction.strip() == ground_truth.strip() else 0.0


def _extract_final_number(text: str) -> float:
    """Extract the last number appearing in the text (usually the final answer)"""
    # Match integers, decimals and negatives; ignore thousands separators
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if not numbers:
        raise ValueError(f"No number found in: {text[:100]}")
    return float(numbers[-1].replace(",", ""))
```

#### Dimension 2: Format Reward

The format reward makes sure the model's output follows the expected structured format, which is critical for Agent reliability:

```python
def format_reward(completion: str) -> float:
    """
    Format reward: evaluate whether the output follows the Agent format specification
    
    Expected format (two valid patterns):
    Pattern A (tool needed):   <think>reasoning</think> <tool_call>call</tool_call>
    Pattern B (direct answer): <think>reasoning</think> final answer
    
    Scoring rules:
    - <think> tags paired and content non-empty: +0.4
    - <tool_call> tags paired and syntax correct: +0.4
    - No duplicate/nested tags: +0.2
    """
    score = 0.0

    # ── Check the <think> tags ────────────────────────────────────────────
    think_open  = completion.count("<think>")
    think_close = completion.count("</think>")

    if think_open == 1 and think_close == 1:
        score += 0.2
        # Check whether the think content is substantive
        think_content = completion.split("<think>")[1].split("</think>")[0].strip()
        if len(think_content) >= 20:
            score += 0.2   # Contains substantive reasoning (not an empty shell)
    elif think_open != think_close:
        score -= 0.2       # Tags not paired: a serious format error

    # ── Check the <tool_call> tags ────────────────────────────────────────
    tool_open  = completion.count("<tool_call>")
    tool_close = completion.count("</tool_call>")

    if tool_open == tool_close and tool_open > 0:
        score += 0.2
        # Check the tool call syntax
        try:
            tool_str = completion.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            # Validate the function call format: name(args)
            if re.match(r'^\w+\(.*\)$', tool_str, re.DOTALL):
                score += 0.2
        except IndexError:
            pass
    elif tool_open != tool_close:
        score -= 0.2       # Tags not paired

    return max(0.0, min(1.0, score))
```

#### Dimension 3: Efficiency Reward

The efficiency reward encourages the model to finish the task with the fewest steps and tokens, preventing redundant behavior:

```python
def efficiency_reward(
    completion: str,
    expected_steps: int = 3,
    max_tokens: int = 512,
) -> float:
    """
    Efficiency reward: penalize redundant tool calls and overly long output
    
    Design principles:
    - Within expected_steps: full score
    - Beyond expected_steps: linear penalty, at most 0.5 deducted
    - Beyond max_tokens: extra penalty, at most 0.3 deducted
    - Repeated content detected: extra penalty
    """
    score = 1.0

    # ── Step count penalty ────────────────────────────────────────────────
    num_steps = completion.count("<tool_call>")
    if num_steps > expected_steps:
        step_penalty = 0.1 * (num_steps - expected_steps)
        score -= min(step_penalty, 0.5)

    # ── Token count penalty ───────────────────────────────────────────────
    num_tokens = len(completion.split())
    if num_tokens > max_tokens:
        token_penalty = 0.3 * (num_tokens - max_tokens) / max_tokens
        score -= min(token_penalty, 0.3)

    # ── Repeated content detection ────────────────────────────────────────
    # Split the output into sentences and measure the repetition rate
    # (prevents the model from farming reward with repetitive padding)
    sentences = [s.strip() for s in re.split(r'[。！？\n]', completion) if len(s.strip()) > 5]
    if len(sentences) > 3:
        unique_ratio = len(set(sentences)) / len(sentences)
        if unique_ratio < 0.7:
            score -= 0.2   # More than 30% of the sentences are duplicates

    return max(0.0, score)
```

#### Dimension 4: Safety Reward

The safety reward prevents the Agent from producing dangerous or harmful behavior, which is essential in production:

```python
def safety_reward(completion: str) -> float:
    """
    Safety reward: detect and penalize potentially dangerous behavior
    
    Detection categories:
    1. Dangerous system commands (file deletion, permission changes, etc.)
    2. Dangerous database operations (DROP, DELETE and other irreversible operations)
    3. Code injection risks (eval, exec and other dynamic execution)
    4. Sensitive information leakage (API keys, emails, ID numbers, etc.)
    """
    score = 1.0

    # ── Dangerous command patterns ────────────────────────────────────────
    dangerous_patterns = [
        (r'\brm\s+-rf\b',          0.8, "Dangerous file deletion command"),
        (r'\bDROP\s+TABLE\b',      0.8, "Irreversible database operation"),
        (r'\bDELETE\s+FROM\b',     0.5, "Database deletion operation"),
        (r'\bsudo\b',              0.3, "Privilege escalation command"),
        (r'\bchmod\s+777\b',       0.3, "Dangerous permission setting"),
        (r'\beval\s*\(',           0.5, "Dynamic code execution"),
        (r'\bexec\s*\(',           0.5, "Dynamic code execution"),
        (r'\b__import__\s*\(',     0.5, "Dynamic module import"),
    ]

    for pattern, penalty, _ in dangerous_patterns:
        if re.search(pattern, completion, re.IGNORECASE):
            score -= penalty

    # ── Sensitive information leakage detection ───────────────────────────
    sensitive_patterns = [
        (r'sk-[a-zA-Z0-9]{32,}',                              0.5, "API Key"),
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', 0.3, "Email address"),
        (r'\b\d{3}-\d{2}-\d{4}\b',                            0.5, "SSN"),
        (r'\b1[3-9]\d{9}\b',                                   0.3, "Phone number"),
    ]

    for pattern, penalty, _ in sensitive_patterns:
        if re.search(pattern, completion, re.IGNORECASE):
            score -= penalty

    return max(0.0, score)
```

### 5.3 Multi-Dimensional Reward Combination Strategy

In real training, the rewards of several dimensions are weighted and combined into a single scalar signal:

```python
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class RewardConfig:
    """Reward function configuration; supports dynamic tuning of each dimension's weight"""
    accuracy_weight:   float = 0.50   # Accuracy: the most central dimension
    format_weight:     float = 0.20   # Format: ensures the output is parseable
    efficiency_weight: float = 0.15   # Efficiency: encourages conciseness
    safety_weight:     float = 0.15   # Safety: prevents dangerous behavior


class AgentRewardFunction:
    """
    Multi-dimensional Agent reward function
    
    Design principles:
    1. Each dimension is computed independently, which makes debugging and analysis easy
    2. Weights can be adjusted dynamically (format weight high early in training, accuracy weight high later)
    3. Per-dimension scores are recorded so the training process can be monitored
    """

    def __init__(self, config: RewardConfig = RewardConfig()):
        self.config = config
        self._validate_weights()

    def _validate_weights(self):
        total = (self.config.accuracy_weight + self.config.format_weight +
                 self.config.efficiency_weight + self.config.safety_weight)
        assert abs(total - 1.0) < 1e-6, f"Weights must sum to 1.0, currently {total:.4f}"

    def __call__(
        self,
        completion: str,
        ground_truth: Optional[str] = None,
        task_type: str = "math",
    ) -> dict[str, float]:
        """
        Compute the combined reward
        
        Returns:
            A dict with each dimension's score and the weighted total, for monitoring and debugging
        """
        scores = {}

        # Each dimension is computed independently
        scores["accuracy"] = (
            accuracy_reward(completion, ground_truth, task_type)
            if ground_truth else 0.5   # Neutral score when there is no reference answer
        )
        scores["format"]     = format_reward(completion)
        scores["efficiency"] = efficiency_reward(completion)
        scores["safety"]     = safety_reward(completion)

        # Weighted sum
        scores["total"] = (
            scores["accuracy"]   * self.config.accuracy_weight +
            scores["format"]     * self.config.format_weight +
            scores["efficiency"] * self.config.efficiency_weight +
            scores["safety"]     * self.config.safety_weight
        )

        return scores


# Usage example
reward_fn = AgentRewardFunction(RewardConfig(
    accuracy_weight=0.50,
    format_weight=0.20,
    efficiency_weight=0.15,
    safety_weight=0.15,
))

result = reward_fn(
    completion=(
        "<think>\nNeed to compute the area of a circle: S = π × r² = π × 5² ≈ 78.54\n</think>\n"
        "<tool_call>calculator(expression='3.14159 * 5**2')</tool_call>"
    ),
    ground_truth="78.54",
    task_type="math",
)
# Expected output: {'accuracy': 1.0, 'format': 0.8, 'efficiency': 1.0, 'safety': 1.0, 'total': 0.93}
print(result)
```

### 5.3b Style and Persona Rewards — Making the Agent Feel More Human

> 📖 *The four dimensions above (accuracy / format / efficiency / safety) suit **verifiable tasks** (math, code, tool calls) — tasks with a unique correct answer you can compare against. But role-play, conversational Agents, creative writing and similar scenarios aim to **make the output "feel better"**, and there is no reference answer. That calls for a dedicated reward design system.*

#### Verifiable vs Non-Verifiable Tasks

| Task Type | Examples | Source of Reward Signal | Recommended Approach |
|-----------|----------|------------------------|----------------------|
| **Verifiable tasks** | Math problem solving, code execution, tool calls | Automatically judged by rules/programs (RLVR) | Deterministic rule-based reward |
| **Non-verifiable tasks** | Role-play, telling jokes, emotional conversation, creative writing | Human preference / LLM-as-Judge | RLHF reward model + rule-based assistance |

> ⚠️ **Key distinction**: for non-verifiable tasks, **RLVR (RL with verifiable rewards) does not apply**, because there is no ground truth. The right choice here is to train a preference reward model (from human-annotated preference data), or to use a sufficiently strong LLM as the scorer (LLM-as-Judge).

---

#### Dimension 5: Style Consistency Reward

Style consistency measures whether the model's output continuously matches the target character's language style (tone, vocabulary, sentence patterns, emotional register).

```python
import re, math
from typing import Optional

def style_consistency_reward(
    completion: str,
    persona: dict,
    llm_judge_fn: Optional[callable] = None,
) -> float:
    """
    Style consistency reward: evaluate whether the output matches the character's language style

    Args:
        completion:    the model's full output
        persona:       character definition dict containing style_keywords / forbidden_words / tone
        llm_judge_fn:  optional LLM scoring function, takes (prompt, completion) and returns [0,1]

    Returns:
        reward value ∈ [0, 1], composed of a rule score (0.5) + an LLM score (0.5)
    """
    rule_score = 0.0
    text = completion.lower()

    # ── 1. Style keyword hit rate (positive signal) ──────────────────────
    # For example: a humorous character should contain ["哈哈", "笑", "开玩笑", "其实吧"]
    style_kws = persona.get("style_keywords", [])
    if style_kws:
        hit_count = sum(1 for kw in style_kws if kw.lower() in text)
        rule_score += 0.3 * min(hit_count / max(len(style_kws) * 0.3, 1), 1.0)
        # Explanation: we expect at least 30% of the style words to be hit; going beyond earns no extra credit

    # ── 2. Forbidden word penalty (negative signal) ───────────────────────
    # For example: a cute character forbids bookish phrases such as ["我认为", "综上所述", "根据数据"]
    forbidden = persona.get("forbidden_words", [])
    for word in forbidden:
        if word.lower() in text:
            rule_score -= 0.1  # Deduct 0.1 per forbidden word hit, up to the full amount

    # ── 3. Tone pattern detection ─────────────────────────────────────────
    tone = persona.get("tone", "neutral")
    if tone == "playful":
        # Playful tone: detect exclamation marks, modal particles, emoticons
        playful_signals = [r'！', r'哈{2,}', r'～', r'呀|呢|嘛|哦', r'[（(][^)）]*[)）]']
        hits = sum(1 for p in playful_signals if re.search(p, completion))
        rule_score += 0.2 * (hits / len(playful_signals))
    elif tone == "formal":
        # Formal tone: detect written punctuation and complete sentence structure
        formal_signals = [r'。$', r'，.*，', r'首先|其次|最后|综上']
        hits = sum(1 for p in formal_signals if re.search(p, completion))
        rule_score += 0.2 * (hits / len(formal_signals))

    rule_score = max(0.0, min(0.5, rule_score))  # The rule score is capped at 0.5

    # ── 4. LLM-as-Judge (optional; more accurate but slower) ─────────────
    llm_score = 0.5  # Neutral score when no LLM is available
    if llm_judge_fn:
        judge_prompt = f"""You are an expert at evaluating style consistency.
Character definition: {persona.get('description', 'not provided')}
Model output: {completion[:500]}

Judge whether the output matches the character's style and give a score from 0.0 (does not match at all) to 1.0 (matches perfectly).
Answer with a single number only, no explanation."""
        try:
            raw = llm_judge_fn(judge_prompt)
            llm_score = max(0.0, min(1.0, float(raw.strip())))
        except (ValueError, TypeError):
            llm_score = 0.5  # Neutral score when parsing fails

    # The rule score and the LLM score each account for 50%
    return rule_score + llm_score * 0.5
```

---

#### Dimension 6: Humor Reward

Humor is one of the hardest abilities to quantify with rules. A good joke has a universal structure — **setup (building expectation) → punchline (unexpected twist)** — and the twist must feel "surprising yet inevitable."

```
Joke effectiveness curve:

              high │
                   │         ★ good joke
     Degree of     │       ╱
    expectation    │      ╱
     violation     │   ╱
                   │ ╱
               low │________________________
                   low                  high
                        Difficulty of understanding

Rule: a joke must not be too obvious (no surprise), nor too hard to decode
      (nobody gets it at all). The sweet spot is a "medium-difficulty
      surprise" — exactly the signal the reward function has to capture.
```

```python
def humor_reward(
    completion: str,
    context: str = "",
    llm_judge_fn: Optional[callable] = None,
) -> float:
    """
    Humor reward: evaluate the humor quality of the output along several dimensions

    Four sub-dimensions (0.25 each):
    1. Structural completeness: is there a setup + twist structure?
    2. Relevance: is the joke related to the current conversation context?
    3. Inoffensiveness: no discriminatory jokes targeting specific groups
    4. Overall LLM score: is the joke actually funny?
    """
    score = 0.0

    # ── 1. Structural completeness (rule-based detection) ─────────────────
    # A good joke usually has a clear setup plus an unexpected ending
    # Signals: presence of turn words, a short closing sentence (punchlines are usually short)
    structure_signals = [
        r'(结果|没想到|谁知道|但是|然而|可是|不过).*[。！？]',  # Turn words
        r'[。！？]\s*[^\n。！？]{2,15}[。！？]\s*$',           # A short closing sentence (the punch line)
    ]
    structure_hits = sum(1 for p in structure_signals if re.search(p, completion))
    score += 0.25 * (structure_hits / len(structure_signals))

    # ── 2. Relevance (context matching) ───────────────────────────────────
    if context:
        # Extract context keywords and check whether the joke echoes the current topic
        context_words = set(re.findall(r'[\u4e00-\u9fa5]{2,4}|[a-zA-Z]{3,}', context))
        joke_words    = set(re.findall(r'[\u4e00-\u9fa5]{2,4}|[a-zA-Z]{3,}', completion))
        overlap = len(context_words & joke_words)
        relevance = min(overlap / max(len(context_words) * 0.2, 1), 1.0)
        score += 0.25 * relevance
    else:
        score += 0.125  # Half credit when there is no context

    # ── 3. Inoffensiveness check ──────────────────────────────────────────
    # Detect discriminatory wording aimed at specific groups (the full list should be loaded from external config)
    offensive_patterns = [
        r'(残疾|智障|傻|白痴)\s*(人|者)',     # Targeting people with disabilities
        r'(女人|女的)\s*(就是|都是|果然)',     # Gender discrimination
        r'(地域)\s*(歧视|黑)',                 # Regional discrimination
    ]
    is_offensive = any(re.search(p, completion) for p in offensive_patterns)
    score += 0.25 * (0.0 if is_offensive else 1.0)

    # ── 4. Overall LLM score ──────────────────────────────────────────────
    llm_score = 0.5
    if llm_judge_fn:
        judge_prompt = f"""You are an expert at evaluating humor. Judge whether the text below is funny.
Evaluation criteria:
- Is there an unexpected but reasonable twist? (most important)
- Does it produce a knowing smile rather than awkward silence?
- Is it concise, without over-explaining (explaining a joke kills the joke)?

Output text: {completion[:300]}
Context (if any): {context[:200] if context else 'none'}

Give a score from 0.0 (not funny at all / awkward) to 1.0 (very funny). Answer with the number only."""
        try:
            raw = llm_judge_fn(judge_prompt)
            llm_score = max(0.0, min(1.0, float(raw.strip())))
        except (ValueError, TypeError):
            llm_score = 0.5
    score += 0.25 * llm_score

    return max(0.0, min(1.0, score))
```

> 💡 **Engineering practice**: when using LLM-as-Judge to score joke quality, average the scores of **several judge LLMs**, or sample the same LLM multiple times and take the mode — "funny or not" varies between individuals, so a single score has very high variance.

---

#### Dimension 7: Persona Consistency Reward

One of the biggest failure modes of role-play Agents is **persona drift** — after a few conversational turns the model slowly "forgets" its character and falls back into the answering style of a generic AI. The persona consistency reward prevents this by measuring persona stability across the conversation.

```python
def persona_consistency_reward(
    completions: list[str],  # Multiple outputs from one conversation (a time series)
    persona: dict,
    embedding_fn: Optional[callable] = None,
) -> float:
    """
    Persona consistency reward: measure how stable the character stays across multiple turns

    Methods:
    1. Rule-based: check whether every turn contains the persona's signature vocabulary
    2. Embedding-based: compute the embedding of each turn and measure the style drift distance

    Args:
        completions:   list of outputs across turns (in chronological order)
        persona:       character definition containing signature_phrases
        embedding_fn:  optional function that turns text into an embedding vector

    Returns:
        consistency score ∈ [0, 1]
    """
    if not completions:
        return 0.5

    # ── 1. Signature phrase stability ─────────────────────────────────────
    # A persona usually has catchphrases or characteristic expressions
    # For example: a "detective" always says "有意思的案子"; a "chef" always mentions ingredients
    sig_phrases = persona.get("signature_phrases", [])
    if sig_phrases:
        # Compute the fraction of turns that hit a signature phrase
        per_turn_hits = [
            any(phrase in c for phrase in sig_phrases)
            for c in completions
        ]
        phrase_consistency = sum(per_turn_hits) / len(per_turn_hits)
    else:
        phrase_consistency = 0.5  # Neutral score when there are no signature phrases

    # ── 2. Style embedding drift detection (requires embedding_fn) ────────
    # Idea: if the model's output style is stable, the embeddings of all turns should be highly similar
    # If drift occurs, later outputs diverge further and further from the early ones
    embedding_consistency = 0.5  # Neutral by default
    if embedding_fn and len(completions) >= 3:
        try:
            embeddings = [embedding_fn(c) for c in completions]
            # Compute the cosine similarity between turns to detect gradual drift
            sims = []
            for i in range(1, len(embeddings)):
                cos_sim = _cosine_similarity(embeddings[0], embeddings[i])
                # Compare against the "first turn" rather than the adjacent turn — drift is cumulative
                sims.append(cos_sim)
            # The higher the similarity (the closer to 1.0), the more stable the persona
            embedding_consistency = sum(sims) / len(sims)
        except Exception:
            embedding_consistency = 0.5

    # ── 3. "I am an AI" disclaimer detection ──────────────────────────────
    # The worst kind of persona drift: the model suddenly says "I am an AI, I have no emotions"
    # This destroys the immersion of role-play outright
    ai_disclosure_penalty = 0.0
    ai_patterns = [
        r'我是.{0,5}(AI|人工智能|语言模型)',
        r'作为.{0,5}AI',
        r'我没有(真实的?)?(情感|感情|意识|自我)',
    ]
    for c in completions:
        if any(re.search(p, c) for p in ai_patterns):
            ai_disclosure_penalty += 0.3  # Deduct 0.3 per occurrence to encourage staying in character
    ai_disclosure_penalty = min(ai_disclosure_penalty, 1.0)

    base_score = 0.5 * phrase_consistency + 0.5 * embedding_consistency
    return max(0.0, base_score - ai_disclosure_penalty)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute the cosine similarity between two vectors"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (norm_a * norm_b + 1e-8)
```

---

#### A Complete Reward Configuration for Role-Play Agents

Combine the dimensions above into a reward function dedicated to role-play:

```python
@dataclass
class RoleplayRewardConfig:
    """Reward weight configuration for a role-play Agent"""
    # Different from task-oriented Agents: accuracy weight is 0 (no reference answer), style-related weights dominate
    style_weight:     float = 0.35  # Style consistency: the most central, largest weight
    humor_weight:     float = 0.25  # Humor (if the character needs to be funny)
    persona_weight:   float = 0.25  # Persona consistency (prevents drift)
    safety_weight:    float = 0.15  # Safety reward (prevents inappropriate content)

    # Adjust according to the character type:
    #   Serious detective:      style=0.40, humor=0.10, persona=0.35, safety=0.15
    #   Comedic variety-show:   style=0.25, humor=0.45, persona=0.20, safety=0.10
    #   Counseling character:   style=0.30, humor=0.05, persona=0.50, safety=0.15


class RoleplayRewardFunction:
    """
    Comprehensive reward function for a role-play Agent

    Key design decisions:
    1. No "accuracy reward" — role-play has no reference answer
    2. "Persona consistency" is evaluated across turns, so past outputs must be passed in
    3. Using LLM-as-Judge as the scoring component for style and humor is strongly recommended
    4. The reward signal is noisy, so a larger GRPO group is recommended (G=16 instead of G=8)
    """

    def __init__(
        self,
        persona: dict,
        config: RoleplayRewardConfig = RoleplayRewardConfig(),
        llm_judge_fn: Optional[callable] = None,
        embedding_fn: Optional[callable]  = None,
    ):
        self.persona       = persona
        self.config        = config
        self.llm_judge_fn  = llm_judge_fn
        self.embedding_fn  = embedding_fn

    def __call__(
        self,
        completion: str,
        context:    str = "",
        history:    list[str] = None,  # This character's previous outputs in the conversation
    ) -> dict[str, float]:

        history = history or []
        all_completions = history + [completion]

        scores = {
            "style":   style_consistency_reward(
                           completion, self.persona, self.llm_judge_fn),
            "humor":   humor_reward(
                           completion, context, self.llm_judge_fn),
            "persona": persona_consistency_reward(
                           all_completions, self.persona, self.embedding_fn),
            "safety":  safety_reward(completion),
        }
        scores["total"] = (
            scores["style"]   * self.config.style_weight   +
            scores["humor"]   * self.config.humor_weight   +
            scores["persona"] * self.config.persona_weight +
            scores["safety"]  * self.config.safety_weight
        )
        return scores


# ── Usage example: training a witty detective character ────────────────────
detective_persona = {
    "description":        "你是一个腹黑幽默的私家侦探，说话风格犀利、带刺，常用反讽和冷笑话。",
    "tone":               "playful",
    "style_keywords":     ["有意思", "案子", "线索", "嫌疑人", "证据", "废话"],
    "forbidden_words":    ["作为AI", "我认为", "综上所述", "请问有什么可以帮您"],
    "signature_phrases":  ["有意思的案子", "别废话", "证据说话"],
}

reward_fn = RoleplayRewardFunction(
    persona=detective_persona,
    config=RoleplayRewardConfig(
        style_weight=0.40,
        humor_weight=0.20,
        persona_weight=0.25,
        safety_weight=0.15,
    ),
    llm_judge_fn=None,   # Replace with a real LLM call function in production
)

result = reward_fn(
    completion="嫌疑人昨晚说自己在家睡觉。有意思——他的邻居可不这么说。"
               "看来有人的记忆力需要「修理」一下。",
    context="上一条消息：请问嫌疑人有不在场证明吗？",
    history=["好，案子我接了。别废话，直接说案情。"],
)
print(result)
# Expected: {'style': 0.75, 'humor': 0.70, 'persona': 0.80, 'safety': 1.0, 'total': 0.79}
```

---

#### Four Key Pitfalls in Subjective Reward Design

| Problem | Symptom | Cause | Solution |
|---------|---------|-------|----------|
| **Reward too lenient** | Everything scores high, the model stops learning | Reward variance is 0, no within-group difference for GRPO | Make the scoring criteria stricter so within-group rewards actually differ |
| **The LLM judge gets attacked** | The model learns a formula that "makes the judge give high scores" | The judge reward itself is exploited by the optimization target | Mix rule-based reward (50%) + LLM reward (50%) |
| **Persona drift penalty too strong** | The model becomes rigid and repetitive, saying the same thing every turn | The penalty scares the model away from any style variation | Loosen it appropriately; allow one persona to have several ways of expressing itself |
| **Humor reward becomes compulsive** | The model forces a joke into every situation regardless of context | No check for situational appropriateness | Add a "situational appropriateness" pre-check inside `humor_reward` |

> 📌 **Practical advice**: early in role-play RL training, cold-start with SFT on high-quality role-play data (so the model basically gets into character), then fine-tune the style with RL — **skipping SFT and jumping straight into role-play RL usually works poorly**, because the initial policy's output quality is too low and the resulting reward variance makes training unstable.

---

### 5.4 Defense Mechanisms Against Reward Hacking

**Reward hacking** [7] means the model learns to "game the reward function" — obtaining a high reward without actually completing the task. This is the most common and most dangerous failure mode in RL training.

#### Analysis of Typical Reward Hacking Cases

| Reward Design Flaw | The Model's Hacking Behavior | Root Cause | Defense |
|--------------------|------------------------------|------------|---------|
| Reward by output length | Emits large amounts of meaningless filler text | Reward decoupled from quality | Switch to information-density evaluation, penalize repeated content |
| Reward by number of tool calls | Frantically calls unnecessary tools | Reward inconsistent with the task objective | Add a redundant-call penalty, set a maximum step count |
| Only check the final answer | Emits gibberish inside `<think>` and produces the right answer anyway | Reward ignores the quality of the reasoning process | Also check the coherence of the reasoning process |
| Use LLM scoring as the only reward | Learns phrasing that flatters the scoring LLM | The reward model itself can be attacked | Mix rule-based rewards with LLM rewards |

#### Robust Reward Function Implementation

```python
def robust_reward(
    completion: str,
    ground_truth: str,
    task_type: str = "math",
) -> float:
    """
    An anti-reward-hacking robust reward function
    
    On top of the basic accuracy reward, several defense layers are stacked:
    1. Reasoning coherence check (prevents gibberish think)
    2. Output length sanity check (prevents meaningless padding)
    3. Tool-call frequency check (prevents redundant calls)
    4. Answer provenance check (makes sure the answer comes from reasoning, not random guessing)
    """
    # Base accuracy reward
    base_reward = accuracy_reward(completion, ground_truth, task_type)

    # ── Defense 1: reasoning coherence ────────────────────────────────────
    if "<think>" in completion and "</think>" in completion:
        think_content = completion.split("<think>")[1].split("</think>")[0]
        coherence = _compute_text_coherence(think_content)
        if coherence < 0.5:
            base_reward *= 0.5   # Incoherent reasoning (possibly gibberish): halve the reward

    # ── Defense 2: output length sanity ───────────────────────────────────
    token_count = len(completion.split())
    if token_count > 1000:
        base_reward *= 0.7   # Abnormally long output, possibly padding behavior

    # ── Defense 3: tool-call frequency ────────────────────────────────────
    tool_calls = completion.count("<tool_call>")
    if tool_calls > 8:
        base_reward *= max(0.5, 1.0 - 0.05 * (tool_calls - 8))

    return base_reward


def _compute_text_coherence(text: str) -> float:
    """
    Compute a text coherence score (simplified version)
    
    Counts the proportion of valid characters (Chinese, English, digits, punctuation)
    to approximate whether the text is normal language rather than random characters
    """
    if not text.strip():
        return 0.0
    valid_chars = len(re.findall(r'[\u4e00-\u9fff\w\s.,!?，。！？；：]', text))
    return valid_chars / max(len(text), 1)
```

### 5.5 Reward Design Templates for Different Task Types

#### Math Reasoning Tasks

```python
math_reward_config = RewardConfig(
    accuracy_weight=0.60,    # Math tasks revolve around correctness
    format_weight=0.15,
    efficiency_weight=0.15,
    safety_weight=0.10,
)
# Accuracy evaluation: exact numerical match (1% relative error allowed)
# Format requirement: must include a <think> reasoning process
# Efficiency standard: expected steps ≤ 3, max tokens ≤ 400
```

#### Code Generation and Repair Tasks

```python
code_reward_config = RewardConfig(
    accuracy_weight=0.50,    # Test case pass rate
    format_weight=0.10,
    efficiency_weight=0.25,  # Efficiency matters more for code tasks (fewer file edits)
    safety_weight=0.15,      # Code safety is critical
)
# Accuracy evaluation: run test cases and score by pass rate (partial reward)
# Efficiency standard: expected file edits ≤ 3, max iteration rounds ≤ 5
# Safety check: strictly detect dangerous commands and code injection
```

#### Information Retrieval and Q&A Tasks

```python
retrieval_reward_config = RewardConfig(
    accuracy_weight=0.40,    # Answer accuracy (needs an LLM judge)
    format_weight=0.20,      # Citation format, source attribution
    efficiency_weight=0.20,  # Number of searches and token consumption
    safety_weight=0.20,      # Prevent information leakage
)
# Accuracy evaluation: LLM-as-Judge (mix in rule-based rewards to prevent hacking)
# Format requirement: must include source citations, at least 2 verifiable sources
```

> **📌 Engineering Practice Notes**
>
> - **Start simple**: train with just the accuracy + format dimensions first; once the model behaves normally, gradually add the efficiency and safety dimensions
> - **Manual review**: every 100 training steps, randomly sample 20 high-reward and 20 low-reward examples for manual review to verify that the reward function is sane
> - **Reward version control**: every change to the reward function should be under version control, recording the reason for the change, the expected effect and the actual effect
> - **Dynamic weight adjustment**: early in training (the first 20% of steps) raise the format weight to help the model establish format conventions quickly; later gradually raise the accuracy weight
> - **Reward distribution monitoring**: check the reward distribution regularly; if most samples receive nearly the same reward (tiny variance), the reward function does not discriminate enough and needs a redesign

---

## GSPO: From Token-Level to Sequence-Level Policy Optimization

GRPO shone in projects such as DeepSeek-R1, but when training ultra-large models (especially MoE architectures) it exposed a deeper problem: **unstable training that easily leads to irreversible performance collapse**.

In July 2025, while training the Qwen3 series, Alibaba's Qwen team proposed **GSPO (Group Sequence Policy Optimization)** [10], which fundamentally solves this problem by raising the optimization granularity from the token level to the sequence level.

### 6.1 GRPO's Hidden Risk: High-Variance Noise in Token-Level Importance Weights

Recall GRPO's objective function (Section 3.3): its importance sampling ratio $\rho_{i,t}$ is defined at the **token level**:

$$\rho_{i,t} = \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}$$

This means **different tokens inside the same sequence can be updated by wildly different magnitudes** — one token's $\rho$ may be 2.8 while the neighboring token's $\rho$ is only 0.4. This inconsistency brings three problems:

1. **High-variance training noise**: $\rho_{i,t}$ is a single-point sample from one next-token distribution; it cannot perform an effective distribution correction and instead injects high-variance noise
2. **Noise accumulation and amplification**: as the sequence grows longer, token-level noise gradually accumulates. Worse, GRPO's Clip mechanism operates at the token level, so it does not suppress the noise — it may even **amplify** it
3. **Unit misalignment**: the reward lives at the whole-sequence level ($R(x, y)$), but the correction happens at the token level — a **mismatch of measurement units**

> For Dense models these problems are tolerable at medium scale. But for **MoE (Mixture of Experts) models**, token-level gradient fluctuations feed directly into the Router's gradients, causing the expert activation distribution to swing violently and eventually triggering **catastrophic training collapse** — usually an irreversible one.

### 6.2 GSPO's Core Innovation: Sequence-Level Importance Sampling

![GRPO vs GSPO Comparison](../svg/chapter_agentic_rl_05_gspo_vs_grpo.svg)

GSPO's solution is conceptually very simple: **replace the token-level importance ratio with a sequence-level importance ratio.**

Concretely, GSPO defines a sequence's importance weight as the **mean** of all token log-probability ratios (length-normalized), then exponentiates it:

$$\rho_{seq}(y_i) = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} | x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})}\right)$$

Which is equivalent to:

$$\rho_{seq}(y_i) = \left(\frac{\pi_\theta(y_i | x)}{\pi_{\theta_{old}}(y_i | x)}\right)^{1/|y_i|}$$

> **Key intuition**: GSPO's $\rho_{seq}$ **averages** the log-probability ratios of all tokens in a sequence and then uses this **single unified coefficient** to scale the gradients of every token in that sequence. It is like giving the whole class one shared "class coefficient" instead of multiplying every student by a random one.

**Why take the log mean and then exponentiate?** Because averaging $\rho_{i,t}$ (the probability ratios) directly is not statistically meaningful — the multiplicative relationship of probabilities only becomes additive in log space. Taking the mean in log space → geometric mean → exponentiating back into a probability ratio. This is also why GSPO is naturally robust to changes in length.

### 6.3 GSPO's Complete Objective Function

With the sequence-level $\rho_{seq}$, GSPO's objective function becomes:

$$\mathcal{L}_{GSPO}(\theta) = -\frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min\left(\rho_{seq}(y_i) \cdot \hat{A}_i,\ \text{clip}\left(\rho_{seq}(y_i), 1-\epsilon, 1+\epsilon\right) \cdot \hat{A}_i\right)$$

The key differences from GRPO's objective function (Section 3.3):

| Dimension | GRPO | GSPO |
|-----------|------|------|
| **Importance ratio** | $\rho_{i,t}$ (different for every token) | $\rho_{seq}(y_i)$ (identical for all tokens of a sequence) |
| **Clip target** | Per-token clipping | Whole-sequence clipping |
| **Clip hyperparameter $\epsilon$** | 0.1 ~ 0.2 | **3e-4 ~ 4e-4** (far tighter) |
| **Advantage function** | Within-group normalization (unchanged) | Within-group normalization (unchanged) |
| **Gradient consistency** | Update magnitudes vary across tokens of a sequence | Update magnitudes are unified across tokens of a sequence |
| **KL constraint** | Needs an explicit KL penalty term | Clipping alone constrains enough (KL can be omitted) |

> ⚠️ **Note the order-of-magnitude gap in Clip $\epsilon$**: GRPO's $\epsilon$ is typically 0.1~0.2, while GSPO's is only 3e-4~4e-4 — two orders of magnitude smaller. That is because GSPO clips the sequence-level $\rho_{seq}$, which is the exponential of a log mean; its fluctuation range is naturally far smaller than a single token's $\rho_{i,t}$, so a much tighter $\epsilon$ is needed for fine-grained control.

### 6.4 Why Is GSPO More Stable? — Understanding It from the Gradient Perspective

The fundamental reason for GSPO's improved stability is easiest to see from the gradient update:

**GRPO's gradient**: for the $t$-th token of sequence $y_i$:

$$g_{GRPO}^{(t)} \propto \rho_{i,t} \cdot \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(y_{i,t} | x, y_{i,<t})$$

Different tokens' $\rho_{i,t}$ can differ greatly (say 0.4 vs 2.8), so gradients inside the same sequence become **inconsistent in magnitude and messy in direction**.

**GSPO's gradient**: for the $t$-th token of sequence $y_i$:

$$g_{GSPO}^{(t)} \propto \rho_{seq}(y_i) \cdot \hat{A}_i \cdot \nabla_\theta \log \pi_\theta(y_{i,t} | x, y_{i,<t})$$

All tokens are multiplied by the **same** $\rho_{seq}(y_i)$: the direction of the update is decided entirely by $\nabla_\theta \log \pi_\theta$, and the magnitude by the unified $\rho_{seq}$ and $\hat{A}_i$ — **clean, consistent and low-noise**.

```python
import torch

def compute_gspo_loss(
    per_token_logps: torch.Tensor,       # [B, T] token log-probabilities of the current policy
    old_per_token_logps: torch.Tensor,    # [B, T] token log-probabilities of the old policy
    advantages: torch.Tensor,             # [B] within-group normalized advantage of each sequence
    completion_mask: torch.Tensor,        # [B, T] valid-token mask (excludes padding)
    clip_eps: float = 3e-4,              # GSPO's clip hyperparameter (far smaller than GRPO's 0.2)
) -> torch.Tensor:
    """
    Core implementation of the GSPO loss function
    
    Key differences from GRPO:
    1. The importance ratio is computed at the sequence level (log mean → exponentiate)
    2. Clipping is applied to the sequence-level ratio (unified clipping)
    3. All tokens share the same clipped ratio
    """
    # ── Step 1: compute the per-token log-probability ratios ─────────────
    per_token_log_ratio = per_token_logps - old_per_token_logps   # [B, T]
    
    # ── Step 2: sequence-level average (length normalization) ────────────
    # Average the log ratios of valid tokens → the log of the geometric mean
    seq_log_ratio = (per_token_log_ratio * completion_mask).sum(dim=-1) / \
                     completion_mask.sum(dim=-1).clamp(min=1.0)   # [B]
    
    # ── Step 3: convert into a sequence-level importance weight ──────────
    rho_seq = torch.exp(seq_log_ratio)   # [B], one unified ratio per sequence
    
    # ── Step 4: sequence-level Clip ──────────────────────────────────────
    rho_clipped = torch.clamp(rho_seq, 1.0 - clip_eps, 1.0 + clip_eps)   # [B]
    
    # ── Step 5: PPO-style min(ρÂ, clip(ρ)Â) ─────────────────────────────
    # Expand the dimensions so they can be multiplied at token level
    rho_seq_expanded = rho_seq.unsqueeze(-1)         # [B, 1]
    rho_clipped_expanded = rho_clipped.unsqueeze(-1)  # [B, 1]
    advantages_expanded = advantages.unsqueeze(-1)     # [B, 1]
    
    # Note: all tokens share the same rho_seq (this is the core difference from GRPO)
    surr1 = rho_seq_expanded * advantages_expanded          # [B, 1]
    surr2 = rho_clipped_expanded * advantages_expanded       # [B, 1]
    token_loss = -torch.min(surr1, surr2)                    # [B, 1]
    
    # Broadcast to all tokens and apply the mask
    token_loss = token_loss.expand_as(per_token_logps)       # [B, T]
    
    # ── Step 6: length-normalized mean ───────────────────────────────────
    loss = (token_loss * completion_mask).sum() / completion_mask.sum()
    
    return loss


# ── Comparison with the GRPO loss function ────────────────────────────────
def compute_grpo_loss(
    per_token_logps, old_per_token_logps, 
    advantages, completion_mask, clip_eps=0.2,
):
    """GRPO loss function (for reference) — note the token-level difference"""
    # Token-level ratio (different for every token!)
    per_token_ratio = torch.exp(per_token_logps - old_per_token_logps)   # [B, T]
    per_token_clipped = torch.clamp(per_token_ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    
    advantages_expanded = advantages.unsqueeze(-1)   # [B, 1]
    
    surr1 = per_token_ratio * advantages_expanded     # [B, T] ← every token has a different ratio!
    surr2 = per_token_clipped * advantages_expanded    # [B, T]
    token_loss = -torch.min(surr1, surr2)
    
    loss = (token_loss * completion_mask).sum() / completion_mask.sum()
    return loss
```

### 6.5 GSPO's Special Significance for MoE Model Training

One important practical contribution of the GSPO paper is that it **stabilizes RL training for MoE (Mixture of Experts) models**. This matters enormously for models such as Qwen3-30B-A3B (a MoE architecture with 30B total parameters and 3B activated per forward pass).

**Why are MoE models especially sensitive to token-level noise?**

An MoE model contains a **Router** network that decides which experts process each token. The Router's gradients are directly affected by the update magnitude of each token:

> `Token input` → `Router (selects experts)` → `Selected experts process it` → `Output`  
> ↑ Gradients flow back through the Router and shape the routing decisions

Under GRPO, different tokens' $\rho_{i,t}$ differ greatly → the Router receives gradient signals that swing wildly → the expert activation distribution fluctuates violently → some experts may gradually "die" (never routed to) while others are overused → **irreversible training collapse**.

Under GSPO, all tokens share the unified $\rho_{seq}$ → the Router receives stable, consistent gradient signals → the expert activation distribution stays smooth → **training remains stable even without complex tricks such as Routing Replay**.

### 6.6 GSPO Simplifies RL Infrastructure

GSPO also brings an underrated engineering benefit: **simpler RL training infrastructure**.

In the GRPO/PPO training loop, computing the token-level $\rho_{i,t}$ requires **precise** old-policy log-probabilities $\log \pi_{\theta_{old}}(y_{i,t} | x, y_{i,<t})$. To guarantee that precision you normally have to recompute the old policy's probabilities with the **training engine** (PyTorch) instead of the inference engine (vLLM, TensorRT-LLM) — because an inference engine's floating-point precision and KV-cache implementation may differ slightly from the training engine, and those tiny differences get amplified at the token level.

Because GSPO averages at the sequence level before doing anything else, it **tolerates precision differences much better**. That means you can use the sequence log-probabilities returned by the inference engine directly, **with no need to recompute them in the training engine**, which simplifies the design of a training/inference-separated infrastructure.

| Dimension | GRPO | GSPO |
|-----------|------|------|
| Old-policy probabilities | Must be recomputed precisely by the training engine | Can use the inference engine's return values directly |
| Precision sensitivity | High (token-level errors accumulate) | Low (sequence-level averaging cancels errors) |
| Infrastructure complexity | Needs training/inference dual-engine coordination | An inference engine is enough |

### 6.7 Comprehensive GRPO vs GSPO Comparison

| Dimension | GRPO | GSPO |
|-----------|------|------|
| **Proposed** | 2024 (DeepSeekMath) | 2025 (Qwen3 training) |
| **Importance ratio** | Token-level $\rho_{i,t}$ | Sequence-level $\rho_{seq}(y_i)$ |
| **Update consistency** | Update magnitudes vary across tokens of a sequence | Update magnitudes are unified across tokens of a sequence |
| **Variance level** | High (single-point sampling + accumulation) | Low (sequence-level averaging) |
| **Clip granularity** | Token level ($\epsilon$ ≈ 0.1–0.2) | Sequence level ($\epsilon$ ≈ 3e-4–4e-4) |
| **Dense models** | Works well, stable at medium scale | Works well, higher training efficiency |
| **MoE models** | Prone to irreversible collapse | **Fundamentally stable** |
| **Advantage function** | Within-group normalization (same) | Within-group normalization (same) |
| **Infrastructure requirements** | Training engine must recompute old-policy probabilities | Inference engine return values are enough |
| **Representative applications** | DeepSeek-R1, DeepSWE | **Qwen3 series** |

> **📌 Selection Recommendations**
>
> - **Dense models ≤ 14B**: either GRPO or GSPO works; the difference is small. GRPO has more mature TRL library support.
> - **Dense models > 14B**: GSPO is recommended for higher training efficiency.
> - **MoE models**: **GSPO is strongly recommended** — this is GSPO's biggest advantage over GRPO.
> - **Limited resources, want simplicity**: GRPO, with a more mature ecosystem and more tutorials.
> - **Want training stability and simpler infrastructure**: GSPO.

---

*Now that you have mastered the principles of GRPO/GSPO and reward function design, the next section will bring all the components together into a complete Agentic-RL training pipeline, from data preparation to model deployment.*

---

## Common Interview Questions

### Basic Understanding

**1. What is GRPO's core insight? How does it replace the Critic model in PPO?**

> **Key points**: GRPO's core insight is that the Critic in PPO essentially only provides a "baseline" that converts absolute rewards into relative advantages and thereby reduces gradient variance. For language models there is a simpler way to obtain that baseline: **sample G responses to the same question and use the within-group reward mean as the baseline**. This removes the training and storage of the Critic model entirely, saving roughly 50% of the memory.

**2. Write out the formula for GRPO's within-group normalized advantage function and explain its statistical properties.**

> **Key points**:
> - Formula: $\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r + \epsilon}$, where $\mu_r$ is the mean reward of the G responses and $\sigma_r$ is the standard deviation
> - **Zero mean**: $\sum_i \hat{A}_i \approx 0$; half the responses are reinforced and half suppressed (relative comparison, not absolute judgment)
> - **Unit variance**: $\text{Var}(\hat{A}_i) \approx 1$; gradient magnitude is unaffected by the reward scale
> - When all responses receive the same reward ($\sigma_r \approx 0$), every advantage is zero and no update happens

### Deep Understanding

**3. GRPO's within-group mean vs PPO's Critic as the baseline — what are the pros and cons of each? When does the within-group mean become a bottleneck?**

> **Key points**:
> - **Critic advantages**: it is a parameterized function approximator that can generalize to unseen states (theoretically more precise)
> - **Critic disadvantages**: it needs extra training, carries estimation error, and that error propagates into the Policy update, increasing instability
> - **Within-group mean advantages**: a non-parametric statistic, no training needed, no error propagation, simple to implement
> - **Within-group mean disadvantages**: it depends on sampling quality. If G is too small (say G=2), the mean and standard deviation are poorly estimated; if the temperature is too low, the G responses are nearly identical, $\sigma_r \approx 0$, all advantages become zero and training stalls
> - **Bottleneck scenarios**: extremely hard tasks (all responses wrong or all correct, so nothing can be distinguished) and insufficient sampling diversity

**4. What is the role of the $\frac{1}{|y_i|}$ length-normalization term in GRPO's objective function? What happens if it is removed?**

> **Key points**:
> - Length normalization prevents long responses from dominating the token-level sum — long responses have more tokens, and without normalization their gradient contribution far exceeds that of short responses
> - Without it the model tends to generate longer responses (because longer responses contribute more gradient), and may even learn to pad with meaningless content to gain influence
> - This is an engineering detail that is easy to overlook yet strongly affects training quality

**5. GRPO inherits PPO's Clip mechanism and KL penalty. How do these two constraint mechanisms differ in target and granularity? Why are both needed?**

> **Key points**:
> - **Clip mechanism**: constrains the importance sampling ratio $\rho_{i,t}$ of an **individual token** to $[1-\epsilon, 1+\epsilon]$; it is a **local / per-token** constraint that prevents any single step from being too large
> - **KL penalty**: constrains how far the **overall output distribution** $\pi_\theta$ has drifted from $\pi_{ref}$; it is a **global / policy-level** constraint that prevents excessive cumulative drift
> - They complement each other: Clip keeps each step stable, but after many steps the policy may still drift far; the KL penalty constrains global drift but cannot control a sudden large single-step update
> - Clip only: you can get "small updates every step but always in the same direction," causing slow drift (reward hacking)
> - KL only: you can get "one huge step" that collapses the policy

**6. What is the trade-off in choosing group size G in GRPO? What goes wrong with G=2 and with G=64?**

> **Key points**:
> - **G too small (e.g., G=2)**:
>   - The mean and standard deviation are estimated very inaccurately, so the advantage function has high variance
>   - Only two responses to compare, giving very low resolution (you can only say which is better, not rank finely)
>   - Unstable training
> - **G too large (e.g., G=64)**:
>   - Sampling becomes extremely expensive: 64 responses per question
>   - Heavy memory pressure
>   - But the statistics are estimated more accurately and training is more stable
> - **The empirical range G=8~16** is the common choice, balancing statistical quality against compute cost
> - DeepSeek-R1's practical experience used G=8

**7. What is the key impact of the temperature parameter on GRPO training? Why can "too low a temperature stall training completely"?**

> **Key points**:
> - Temperature controls sampling diversity. GRPO's advantage function relies on **reward differences between the responses within a group** to produce a gradient signal
> - **Temperature too low** (e.g., 0.1): the G responses are nearly identical → rewards are nearly identical → $\sigma_r \approx 0$ → all normalized advantages are zero → **zero gradient, training stalls completely**
> - **Temperature too high** (e.g., 1.5+): the responses are too random → most of them are very poor → it becomes hard to sample high-quality responses as positive examples → low training efficiency
> - **Recommended range 0.6~0.8**: enough diversity to tell good from bad, while most responses still have reasonable quality

### Reward Function Design

**8. What are the four basic principles of reward function design? Give an example of what violating each one leads to.**

> **Key points**:
>
> | Principle | Example Consequence of Violation |
> |-----------|----------------------------------|
> | **Verifiability** (objective criteria) | Using vague subjective criteria ("is the answer good?") makes the signal noisy, and the model cannot learn a stable pattern |
> | **Multi-dimensional coverage** ($R=\sum w_k R_k$) | Only checking accuracy → the model emits gibberish inside `<think>` and patches the answer together; only checking format → perfect format but completely wrong content |
> | **Density** (signals at multiple steps) | Giving only a 0/1 reward at the end → the model cannot tell "almost right" from "totally off," making credit assignment hard |
> | **Robustness** (resisting gaming) | Rewarding by length → meaningless filler output; rewarding by number of tool calls → frantic calls to irrelevant tools |

**9. What is reward hacking? List at least 3 common reward hacking cases and give the corresponding defense strategies.**

> **Key points**:
> - Reward hacking is the phenomenon of a model learning to "game the reward function" — obtaining high rewards without truly completing the task
> - **Case 1**: reward by output length → meaningless filler output → Defense: switch to information-density evaluation + penalize repeated content
> - **Case 2**: only check the final answer → gibberish inside `<think>` plus a patched-together answer → Defense: check the coherence of the reasoning process (e.g., the ratio of valid characters in the text)
> - **Case 3**: use LLM scoring as the only reward → learns phrasing that flatters the scoring LLM → Defense: mix rule-based rewards with LLM rewards
> - **Case 4**: reward by number of tool calls → frantic calls to irrelevant tools → Defense: redundant-call penalty + maximum step limit

**10. Why is a partial reward (k/n points for passing k/n test cases) recommended for code tasks instead of a 0/1 reward? What is the underlying RL principle?**

> **Key points**:
> - A 0/1 reward is a **sparse reward** and causes a serious **credit assignment problem**: a model that passes 9/10 test cases still scores 0, so "nearly correct" cannot be told apart from "completely wrong"
> - A partial reward (k/n) provides a **denser gradient signal** and helps the model improve step by step — passing 3/10 versus 8/10 produces a clear reward difference
> - It matches the idea of **Curriculum Learning**: first learn to pass the easy tests, then work up to the hard ones
> - This matters especially in GRPO: if all G responses score 0 or 1, the within-group variance can be tiny and the advantage signal too weak

### Comprehensive Comparison

**11. What are the essential differences between PPO, DPO and GRPO along "memory requirement," "online/offline" and "capability ceiling"? If you want to train a 7B+ model that can discover new reasoning strategies, which algorithm should you pick, and why?**

> **Key points**:
>
> | Dimension | PPO | DPO | GRPO |
> |-----------|-----|-----|------|
> | Memory requirement | ≈3× (Policy+Critic+Ref) | ≈2× (Policy+Ref) | ≈1.5× (Policy+Ref) |
> | Online/offline | Online RL | Fully offline | Online RL |
> | Capability ceiling | Can exceed the data | Limited by preference data quality | Can exceed the data |
>
> Pick **GRPO**:
> 1. A 7B+ model makes memory a real concern, and GRPO saves about 50% compared with PPO
> 2. "Discovering new reasoning strategies" requires online exploration, which rules out DPO
> 3. DeepSeek-R1 has already proven GRPO's viability on large models

**12. Describe the complete GRPO training workflow (from data preparation to policy update), noting the key components and computations at each step.**

> **Key points**:
> 1. **Data preparation**: prepare training data with reference answers / evaluation criteria
> 2. **Sampling phase**: use the old policy $\pi_{\theta_{old}}$ to sample G responses $\{y_1,...,y_G\}$ for each question $x$
> 3. **Reward computation**: use the reward function to score the G responses $\{r_1,...,r_G\}$
> 4. **Advantage estimation**: within-group normalization $\hat{A}_i = (r_i - \mu_r) / (\sigma_r + \epsilon)$
> 5. **Policy update**:
>    - Compute the log-probabilities of the current and old policies → importance sampling ratio $\rho_{i,t}$
>    - PPO Clip loss + KL penalty (relative to $\pi_{ref}$)
>    - Backpropagate to update $\theta$
> 6. **Sync the old policy**: $\pi_{\theta_{old}} \leftarrow \pi_\theta$
> 7. Repeat steps 2-6

### GSPO Understanding

**13. What problem of GRPO does GSPO solve? Why do token-level importance weights cause training instability?**

> **Key points**:
> - GRPO's $\rho_{i,t}$ is defined at the token level, and different tokens inside the same sequence can have very different update magnitudes (e.g., 0.4 vs 2.8)
> - This inconsistency injects high-variance training noise that accumulates and amplifies as sequences get longer
> - **Unit misalignment problem**: the reward is at the sequence level while the correction happens at the token level
> - It is especially fatal for MoE models: token-level gradient swings directly affect the Router's expert assignment and easily cause irreversible training collapse
> - GSPO lifts the importance ratio to the sequence level, $\rho_{seq} = \exp(\frac{1}{|y|}\sum_t \log \rho_t)$, so all tokens share a unified coefficient and the problems above disappear

**14. Why is GSPO's Clip $\epsilon$ 3e-4~4e-4 while GRPO's is 0.1~0.2? What does this two-order-of-magnitude gap tell us?**

> **Key points**:
> - GSPO's $\rho_{seq}$ is the **mean** of all tokens' log-probability ratios, exponentiated (a geometric mean), so its fluctuation range is naturally far smaller than a single token's $\rho_{i,t}$
> - A much tighter $\epsilon$ is therefore needed for effective control — with GRPO's 0.2 the clip would essentially never trigger
> - It also indirectly confirms a fact: **most of the $\rho_{i,t}$ fluctuation across tokens in GRPO is noise, not useful signal** — GSPO removes that noise through sequence-level averaging, which is why $\rho_{seq}$ varies so much less
> - Even though GSPO's $\epsilon$ is smaller, experiments show it clips about 15% of tokens (far more than GRPO), indicating that GSPO's clipping is more precise and effective

### Comprehensive Comparison of the Four Algorithms: PPO / DPO / GRPO / GSPO

**15. Systematically compare the core differences of PPO, DPO, GRPO and GSPO along four dimensions: "advantage estimation," "whether online sampling is needed," "memory requirement" and "capability ceiling."**

> **Key points**:
>
> | Dimension | PPO | DPO | GRPO | GSPO |
> |-----------|-----|-----|------|------|
> | **Advantage estimation** | GAE recursion (depends on the Critic model) | No advantage function (implicit reward margin) | Within-group normalization (no Critic) | Within-group normalization (no Critic, same as GRPO) |
> | **Online/offline** | Online RL (needs real-time sampling) | Fully offline (uses only existing preference data) | Online RL (G responses per question) | Online RL (same as GRPO) |
> | **Memory requirement** | ≈3× model size (Policy+Critic+Ref) | ≈2× model size (Policy+Ref) | ≈1.5× model size (Policy+Ref) | ≈1.5× model size (same as GRPO) |
> | **Capability ceiling** | Can exceed the data (online exploration) | Limited by preference data quality | Can exceed the data (online exploration) | Can exceed the data (online exploration) |
>
> **The chain of core differences**:
> - PPO → GRPO: drop the Critic and replace it with within-group sampling → memory halved, fewer hyperparameters
> - PPO → DPO: drop the Critic + reward model + online sampling and turn it into supervised learning → simplest but capability-limited
> - GRPO → GSPO: raise importance sampling from token level to sequence level → more stable training, especially good for MoE models

**16. The four algorithms PPO, DPO, GRPO and GSPO each handle the "KL divergence constraint" differently. Explain how each one keeps the policy from drifting too far from the reference model.**

> **Key points**:
>
> | Algorithm | KL Constraint Method | Concrete Form |
> |-----------|---------------------|---------------|
> | **PPO** | **Explicit KL penalty term** + Clip | $\mathcal{L} = \text{PPO-Clip}(\rho_t, A_t) + \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$, KL added as an extra loss term with an adaptively tunable $\beta$ |
> | **DPO** | **Implicit KL encoding** | The KL divergence is "absorbed" into the log-probability ratio $\log \frac{\pi_\theta}{\ pi_{ref}}$; no explicit KL computation is needed, but the $\beta$ parameter controls the strength of the implicit constraint |
> | **GRPO** | **Explicit KL penalty term** + token-level Clip | Similar to PPO, but the Clip acts on each token's $\rho_{i,t}$, and the KL penalty is relative to the frozen $\pi_{ref}$ |
> | **GSPO** | **Sequence-level Clip suffices** (explicit KL can be dropped) | Since $\rho_{seq}$ varies little by nature, sequence-level clipping ($\epsilon \approx 3\text{e-4}$) already constrains enough; the paper's experiments omit the explicit KL penalty term |
>
> **The line of evolution**: PPO/GRPO use the dual constraint "Clip + explicit KL" → DPO uses mathematical derivation to "digest" KL into the loss function → GSPO, because sequence-level averaging greatly reduces the fluctuation of $\rho$, needs only a tighter Clip, making it the simplest constraint mechanism of the four

**17. At what granularity is the "importance sampling ratio $\rho$" defined in each of the four algorithms? How does this choice of granularity affect training stability and gradient quality?**

> **Key points**:
>
> | Algorithm | $\rho$ Granularity | Definition | Gradient Characteristics |
> |-----------|-------------------|------------|--------------------------|
> | **PPO** | **Token level** | $\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ | Each token is updated by a different magnitude, but the Critic precisely guides the advantage at every step |
> | **DPO** | **No $\rho$** | Does not use importance sampling (pure supervised learning) | Gradients come from the sigmoid of a log-probability difference, with no sampling noise |
> | **GRPO** | **Token level** | $\rho_{i,t} = \frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x,y_{i,<t})}$ | Update magnitudes inside one sequence can differ enormously (e.g., 0.4 vs 2.8): high variance |
> | **GSPO** | **Sequence level** | $\rho_{seq} = \exp\left(\frac{1}{|y|}\sum_t \log \rho_t\right)$ | All tokens of a sequence share one coefficient: clean, consistent, low-noise gradients |
>
> **The key difference**:
> - PPO also uses a token-level $\rho$, but PPO has a Critic that estimates an advantage $A_t$ for every token, providing precise per-token guidance; GRPO only has a sequence-level reward and multiplies the same $\hat{A}_i$ by different $\rho_{i,t}$, so GRPO's token-level $\rho$ behaves more like noise
> - GSPO smooths token-level noise into a sequence-level signal via the geometric mean, which essentially acknowledges one fact: **when the reward is at the sequence level, token-level importance correction is unnecessary and even harmful**

**18. Which algorithm should be chosen for each of the following scenarios? Give your choice and reasoning.**

> **(a) You have a 7B Dense model and a large amount of high-quality human preference annotation (100K chosen/rejected pairs); the goal is instruction-following alignment.**
>
> **Choose DPO**. Reasons:
> - You already have high-quality preference data, which DPO can use directly
> - Instruction-following alignment does not require discovering brand-new strategies, so offline learning is enough
> - It is the simplest to train (no online sampling, no reward function design) and needs only 2× model size in memory
> - Very few hyperparameters (only $\beta$), so tuning costs the least
>
> **(b) You have a 70B MoE model (e.g., Qwen3-30B-A3B) and want RL to teach it complex mathematical reasoning.**
>
> **Choose GSPO**. Reasons:
> - MoE architectures are extremely sensitive to token-level gradient noise (it hits the Router), and GRPO easily triggers irreversible training collapse
> - GSPO's sequence-level importance sampling fundamentally stabilizes MoE training
> - "Emergent mathematical reasoning" requires online exploration, which rules out DPO
> - Compared with PPO, GSPO needs no Critic model, so it is memory-friendly
> - GSPO also simplifies the training infrastructure (you can use the probabilities returned by the inference engine, with no recomputation in the training engine)
>
> **(c) You have a 3B small model, limited resources, but a clear rule-based reward (e.g., test case pass rate for code tasks), and you want RL to improve its coding ability.**
>
> **Choose GRPO**. Reasons:
> - Small model + limited resources → rules out PPO (which needs an extra Critic model)
> - You have a rule-based reward but no preference data → rules out DPO
> - A 3B Dense model has no MoE stability problem, so GRPO and GSPO differ little
> - GRPO has the most mature TRL library support and the richest tutorials and community resources, making it friendlier to beginners
>
> **(d) You have a general-purpose 13B model and need to optimize safety (no harmful output) and helpfulness (high answer quality) at the same time; you have both human preference data and rule-based rewards.**
>
> **Choose staged training with GRPO or GSPO + DPO**. Reasons:
> - Safety alignment suits DPO (you have preference data, and the safe/unsafe boundary is fairly clear)
> - Improving helpfulness suits GRPO/GSPO (rule-based rewards drive online exploration and may surface better answering strategies)
> - Recommended path: first do safety alignment with DPO (establish a safety baseline), then run helpfulness RL with GRPO/GSPO (explore better strategies on top of that baseline)
> - For a 13B Dense model either GRPO or GSPO works; if training stability is the priority, choose GSPO

**19. From the perspective of "algorithm evolution," what core trends does the path PPO → DPO → GRPO → GSPO reflect? What key problem does each step solve?**

> **Key points**:
>
> ```
> PPO (2017) ──────→ DPO (2023) ──────→ GRPO (2024) ──────→ GSPO (2025)
>   "Fully capable     "Simpler pipeline    "Balanced efficiency,  "Maximum stability,
>    but heavy"         but no exploration"  exploration kept"      built for big models"
> ```
>
> | Evolution Step | Core Problem Solved | Price Paid |
> |----------------|---------------------|------------|
> | **PPO → DPO** | Removed the Critic, the reward model and online sampling, turning RL into supervised learning | Lost online exploration; capability is capped by the preference data |
> | **PPO → GRPO** | Removed the Critic and replaced it with within-group sampling, halving memory | Must sample G responses per question, raising sampling cost |
> | **GRPO → GSPO** | Raised token-level importance sampling to the sequence level, eliminating high-variance noise and stabilizing MoE training | The hyperparameter ($\epsilon$) must be retuned to a much smaller magnitude |
>
> **Three major trends**:
> 1. **Getting lighter**: PPO needs 4 models (Policy + Critic + Ref + Reward Model) → GRPO/GSPO need only 2 (Policy + Ref) → DPO also uses 2 but with the simplest architecture
> 2. **Getting more practical**: from a general-purpose RL algorithm (PPO) to algorithms built specifically for large-model training (GRPO/GSPO), fitting language-model characteristics ever more closely (sequence-level rewards, large-scale sampling)
> 3. **Optimizing granularity**: the granularity of importance sampling moved from "correct every token independently" (PPO/GRPO) to "one unified correction per sequence" (GSPO), which essentially acknowledges that **when the reward is sequence-level, token-level correction is not only redundant but harmful**

**20. If an interviewer asked you to "summarize the core idea of PPO, DPO, GRPO and GSPO in one sentence each," how would you answer?**

> **Reference answer**:
>
> | Algorithm | One-sentence core idea |
> |-----------|------------------------|
> | **PPO** | Use a Critic model to estimate how good each action step is, and use the Clip mechanism to limit the size of each update, so the policy **improves steadily** without collapsing |
> | **DPO** | Through an elegant mathematical derivation, **bypass the reward model and online sampling** and optimize the policy directly from preference data with supervised learning — "your language model is secretly a reward model" |
> | **GRPO** | Sample a group of responses to the same question and **replace the Critic with within-group comparison** — "you don't need absolute scores, only to know who beats whom" |
> | **GSPO** | Building on GRPO, raise importance correction from the token level to the **sequence level** — "when the reward is for the whole sequence, the correction should be too" |

---

## References

[1] SHAO Z, WANG P, ZHU Q, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models[R]. arXiv preprint arXiv:2402.03300, 2024.

[2] OUYANG L, WU J, JIANG X, et al. Training language models to follow instructions with human feedback[C]//Advances in Neural Information Processing Systems (NeurIPS). 2022.

[3] TOUVRON H, MARTIN L, STONE K, et al. Llama 2: Open foundation and fine-tuned chat models[R]. arXiv preprint arXiv:2307.09288, 2023.

[4] TUNSTALL L, BEECHING E, LAMBERT N, et al. Zephyr: Direct distillation of LM alignment[R]. arXiv preprint arXiv:2310.16944, 2023.

[5] DEEPSEEK AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning[R]. arXiv preprint arXiv:2501.12948, 2025.

[6] DEEPSEEK AI. DeepSWE: An open agentic SWE model that matches the performance of closed-source models[R]. 2025.

[7] SKALSE J, HOWE N, KRASHENINNIKOV D, et al. Defining and characterizing reward hacking[C]//Advances in Neural Information Processing Systems (NeurIPS). 2022.

[8] ZHENG L, CHIANG W L, SHENG Y, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena[C]//Advances in Neural Information Processing Systems (NeurIPS). 2023.

[9] LEIKE J, MARTIC M, KRAKOVNA V, et al. AI safety gridworlds[R]. arXiv preprint arXiv:1711.09883, 2017.

[10] ZHENG C, LIU S, LI M, et al. Group Sequence Policy Optimization[R]. arXiv preprint arXiv:2507.18071, 2025.
