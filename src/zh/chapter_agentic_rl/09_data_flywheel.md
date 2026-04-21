# 11.9 Agentic 数据飞轮：让 Agent 自我进化

> 🔄 *"最好的训练数据不是人工标注的，而是你的 Agent 自己产生的。"*

第 11.8 节解决了"如何构建第一批 Agent 训练数据"。但这只是起点。

真正强大的 Agent 系统遵循一个循环：**Agent 运行 → 产生轨迹 → 过滤高质量轨迹 → 训练更强模型 → 更强模型产生更好轨迹 → ...**

这个闭环就是 **Agentic 数据飞轮（Agentic Data Flywheel）**——也是 DeepSeek、OpenAI、Anthropic 等顶尖团队能持续迭代的核心秘密之一。

---

## 飞轮的整体架构

```
                    ┌─────────────────────────────────────────┐
                    │          Agentic 数据飞轮               │
                    └─────────────────────────────────────────┘

  ┌──────────┐     轨迹日志      ┌──────────────┐   高质量轨迹
  │          │ ──────────────►  │   收集层      │ ──────────────►
  │  Agent   │                  │  Collection  │              │
  │  线上    │                  └──────────────┘              │
  │  服务    │                                                │
  │          │ ◄──────────────  ┌──────────────┐   微调后的    │
  └──────────┘     新版本部署   │   训练层      │ ◄──────────── │
                                │  Training    │              │
                                └──────────────┘              │
                                       ▲                      │
                                       │ 过滤后的数据          │
                                ┌──────────────┐              │
                                │   过滤层      │ ◄────────────┘
                                │   Filter     │
                                └──────────────┘
                                       ▲
                                       │ 自动验证 + 奖励打分
                                ┌──────────────┐
                                │   评估层      │
                                │  Evaluation  │
                                └──────────────┘
```

每一圈飞轮转动，模型能力提升一个台阶；台阶越高，产生的轨迹质量越好，下一圈的原材料更优质。

---

## 第一层：轨迹收集

飞轮的原材料是 Agent 运行时产生的**完整交互记录**（Trajectory）。

### 什么值得收集？

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentTrajectory:
    """一条完整的 Agent 运行轨迹"""
    
    # 基本元数据
    trajectory_id: str
    timestamp: datetime
    session_id: str
    
    # 完整交互记录
    system_prompt: str
    messages: list[dict]         # 完整的 system/user/assistant/tool 序列
    tool_definitions: list[dict] # 本次会话的工具定义
    
    # 结果信息
    final_response: str
    task_completed: bool
    
    # 成本信息（用于后续筛选）
    total_tokens: int
    total_tool_calls: int
    tool_call_failures: int
    latency_ms: float
    
    # 用户反馈（如有）
    user_rating: float | None = None
    user_followup: bool = False   # 用户是否继续追问（隐式满意信号）
    
    # 自动标注（异步计算）
    auto_quality_score: float = 0.0
    is_in_training_set: bool = False


class TrajectoryCollector:
    """线上轨迹收集系统"""
    
    def __init__(self, storage_backend, sampling_rate: float = 1.0):
        self.storage = storage_backend
        self.sampling_rate = sampling_rate  # 采样率（高流量时可<1.0）
    
    async def record(self, trajectory: AgentTrajectory):
        """记录一条轨迹（异步，不阻塞主流程）"""
        import random
        
        # 采样：不是每条都存
        if random.random() > self.sampling_rate:
            return
        
        # 脱敏处理
        cleaned = await self._anonymize(trajectory)
        
        # 异步存储（不影响线上延迟）
        await self.storage.save(cleaned)
        
        # 触发异步质量评分
        asyncio.create_task(self._score_trajectory(cleaned))
    
    async def _anonymize(self, traj: AgentTrajectory) -> AgentTrajectory:
        """脱敏：删除 PII"""
        import re
        for msg in traj.messages:
            content = str(msg.get("content", ""))
            content = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE]', content)
            content = re.sub(r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\b', '[EMAIL]', content)
            msg["content"] = content
        return traj
```

---

## 第二层：质量过滤与标注

原始轨迹质量参差不齐，必须经过严格筛选才能进入训练集。

### 双维度过滤：结果 × 过程

```python
class TrajectoryFilter:
    """
    双维度过滤：
    - 结果维度：任务完成了吗？用户满意吗？
    - 过程维度：推理合理吗？工具使用规范吗？
    """
    
    def __init__(self, reward_model=None):
        self.reward_model = reward_model
    
    def compute_quality_score(self, traj: AgentTrajectory) -> float:
        """综合质量分（0-1）"""
        scores = {}
        
        # ── 结果维度（占 50%）──────────────────────────────
        # 1a. 显式用户满意度
        if traj.user_rating is not None:
            scores["explicit_satisfaction"] = traj.user_rating / 5.0
        else:
            scores["explicit_satisfaction"] = 0.5  # 中性默认值
        
        # 1b. 隐式满意度（用户继续对话 = 满意）
        scores["implicit_satisfaction"] = 0.8 if traj.user_followup else 0.4
        
        # 1c. 任务是否有实质性完成
        scores["task_completion"] = 1.0 if traj.task_completed else 0.2
        
        # ── 过程维度（占 50%）──────────────────────────────
        # 2a. 工具调用成功率
        if traj.total_tool_calls > 0:
            success_rate = 1 - traj.tool_call_failures / traj.total_tool_calls
            scores["tool_success_rate"] = success_rate
        else:
            scores["tool_success_rate"] = 1.0  # 没调用工具也算正常
        
        # 2b. 效率（轨迹长度惩罚，避免空转）
        optimal_calls = max(1, traj.total_tool_calls)
        efficiency = min(1.0, 5 / max(traj.total_tool_calls, 1))
        scores["efficiency"] = efficiency if traj.total_tool_calls <= 5 else 0.5
        
        # 2c. 格式合规性（工具调用都是合法 JSON）
        scores["format_compliance"] = self._check_format_compliance(traj.messages)
        
        # 2d. 奖励模型打分（如果有）
        if self.reward_model:
            rm_score = self.reward_model.score(traj)
            scores["reward_model"] = rm_score
        
        # 加权平均
        weights = {
            "explicit_satisfaction": 0.2,
            "implicit_satisfaction": 0.1,
            "task_completion": 0.2,
            "tool_success_rate": 0.2,
            "efficiency": 0.1,
            "format_compliance": 0.1,
            "reward_model": 0.1 if self.reward_model else 0,
        }
        
        total_weight = sum(weights[k] for k in scores)
        final_score = sum(scores[k] * weights[k] for k in scores) / total_weight
        return final_score
    
    def _check_format_compliance(self, messages: list[dict]) -> float:
        """检查工具调用格式合规率"""
        tool_call_messages = [m for m in messages 
                              if m.get("role") == "assistant" and "tool_calls" in m]
        if not tool_call_messages:
            return 1.0
        
        valid = 0
        for msg in tool_call_messages:
            try:
                for call in msg["tool_calls"]:
                    assert "name" in call and "arguments" in call
                    assert isinstance(call["arguments"], dict)
                valid += 1
            except (AssertionError, KeyError, TypeError):
                pass
        
        return valid / len(tool_call_messages)
    
    def filter_batch(self, 
                     trajectories: list[AgentTrajectory],
                     min_score: float = 0.7) -> list[AgentTrajectory]:
        """批量过滤，返回高质量样本"""
        filtered = []
        score_distribution = []
        
        for traj in trajectories:
            score = self.compute_quality_score(traj)
            traj.auto_quality_score = score
            score_distribution.append(score)
            
            if score >= min_score:
                filtered.append(traj)
        
        # 打印统计信息
        import statistics
        print(f"过滤结果：{len(filtered)}/{len(trajectories)} 通过")
        print(f"质量分布：mean={statistics.mean(score_distribution):.2f}, "
              f"median={statistics.median(score_distribution):.2f}")
        return filtered
```

### 负样本的价值：失败的轨迹也有用

```python
class NegativeSampleExtractor:
    """从失败轨迹中提取对比学习样本"""
    
    def extract_contrastive_pairs(
        self, 
        good_traj: AgentTrajectory,
        bad_traj: AgentTrajectory
    ) -> dict:
        """
        构建偏好对（用于 DPO 训练）
        相同任务，一个成功轨迹（chosen）+ 一个失败轨迹（rejected）
        """
        return {
            "prompt": good_traj.messages[1]["content"],  # 用户问题
            "chosen": self._trajectory_to_text(good_traj),
            "rejected": self._trajectory_to_text(bad_traj),
            "chosen_score": good_traj.auto_quality_score,
            "rejected_score": bad_traj.auto_quality_score,
        }
    
    def extract_error_cases(self, 
                            bad_trajectories: list[AgentTrajectory]) -> list[dict]:
        """
        从失败案例中提取训练样本：
        错误发生点 → 模型应该怎么做（人工修正或强模型修正）
        """
        error_samples = []
        
        for traj in bad_trajectories:
            # 找到第一个出错的位置
            error_point = self._find_first_error(traj.messages)
            if error_point is None:
                continue
            
            # 截取到出错点之前的上下文
            context = traj.messages[:error_point]
            
            # 用强模型生成正确的续写
            correct_continuation = self._get_correct_continuation(context, traj.tool_definitions)
            
            if correct_continuation:
                error_samples.append({
                    "context": context,
                    "wrong_response": traj.messages[error_point],
                    "correct_response": correct_continuation,
                    "error_type": self._classify_error(traj.messages[error_point]),
                })
        
        return error_samples
    
    def _classify_error(self, message: dict) -> str:
        """对错误类型分类（用于分析）"""
        content = str(message.get("content", ""))
        tool_calls = message.get("tool_calls", [])
        
        if tool_calls:
            call_name = tool_calls[0].get("name", "")
            if "not_exist" in call_name.lower() or call_name not in self.available_tools:
                return "hallucinated_tool"
            if not isinstance(tool_calls[0].get("arguments"), dict):
                return "invalid_format"
        
        if "error" in content.lower() and len(content) < 30:
            return "bare_error_propagation"  # 直接把错误消息返回给用户
        
        return "reasoning_error"
```

---

## 第三层：自动标注与奖励模型

对于没有用户反馈的轨迹，需要自动判断质量。

### 基于可验证任务的奖励（RLVR 思路）

```python
class VerifiableRewardCalculator:
    """
    对于有可验证答案的任务，直接验证结果。
    这是最精准的奖励信号，无需奖励模型。
    
    类似于 DeepSeek-R1 的做法：只训练可验证任务，
    奖励来自于答案本身的正确性。
    """
    
    def __init__(self):
        self.verifiers = {
            "math": self._verify_math,
            "code": self._verify_code_execution,
            "sql": self._verify_sql_result,
            "factual": self._verify_factual_claim,
        }
    
    def compute_reward(self, 
                       traj: AgentTrajectory, 
                       task_type: str,
                       ground_truth=None) -> float:
        """计算奖励分数"""
        if task_type not in self.verifiers:
            return None  # 无法自动验证
        
        final_response = traj.final_response
        return self.verifiers[task_type](final_response, ground_truth)
    
    def _verify_math(self, response: str, ground_truth: float) -> float:
        """验证数学答案"""
        import re
        # 提取回复中的数字
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if not numbers:
            return 0.0
        
        predicted = float(numbers[-1])  # 通常最后一个数字是答案
        if abs(predicted - ground_truth) < 1e-6:
            return 1.0
        elif abs(predicted - ground_truth) / (abs(ground_truth) + 1e-9) < 0.05:
            return 0.5  # 近似正确
        return 0.0
    
    def _verify_code_execution(self, response: str, expected_output: str) -> float:
        """提取并执行代码，验证输出"""
        import subprocess, re, tempfile
        
        # 提取代码块
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        if not code_match:
            return 0.0
        
        code = code_match.group(1)
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                fname = f.name
            
            result = subprocess.run(
                ['python', fname], 
                capture_output=True, text=True, timeout=5
            )
            
            actual_output = result.stdout.strip()
            return 1.0 if actual_output == expected_output.strip() else 0.0
            
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception:
            return 0.0


class LLMRewardModel:
    """
    当任务不可自动验证时，用强 LLM 作为奖励模型。
    
    成本权衡：每条轨迹调用一次 GPT-4.1 评估。
    建议：只对边界分数（0.5-0.7）的样本调用奖励模型。
    """
    
    EVALUATION_PROMPT = """你是一个 Agent 轨迹质量评估专家。

任务描述：{task}

Agent 的完整轨迹：
{trajectory}

请从以下维度评估（1-5分）：
1. **推理质量**：思维过程是否清晰、合理？
2. **工具使用**：是否恰当选择并正确调用了工具？
3. **错误处理**：遇到错误时是否优雅处理？
4. **回答质量**：最终回答是否准确、完整？
5. **效率**：是否避免了不必要的步骤？

输出 JSON 格式：
{{"scores": [score1, score2, score3, score4, score5], "overall": <0-1的浮点数>, "reason": "<简短说明>"}}"""
    
    async def score(self, traj: AgentTrajectory) -> float:
        """用 GPT-4.1 对轨迹打分"""
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        
        trajectory_text = self._format_trajectory(traj)
        
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",  # 用便宜的模型降低成本
            messages=[{
                "role": "user",
                "content": self.EVALUATION_PROMPT.format(
                    task=traj.messages[1]["content"],
                    trajectory=trajectory_text
                )
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        
        result = json.loads(response.choices[0].message.content)
        return result.get("overall", 0.5)
```

---

## 第四层：训练与迭代

### 飞轮的迭代节奏

```
┌─────────────────────────────────────────────────────────────────┐
│                     典型迭代节奏（生产团队）                      │
├──────────┬──────────────────────────────────────────────────────┤
│ 第1周    │ 基础版 Agent 上线，收集 ~5000 条轨迹                  │
│ 第2周    │ 过滤 → 获得 ~3000 条高质量样本 → 微调 v0.2           │
│ 第3-4周  │ v0.2 产生更好轨迹（2x 成功率）→ 积累 10000 条        │
│ 第5-6周  │ 过滤 + DPO（加入失败轨迹对比）→ 微调 v0.3            │
│ 第7-8周  │ v0.3 显著好于 v0.1，继续收集...                      │
│ ...      │ 每 2-4 周一个版本，能力持续提升                       │
└──────────┴──────────────────────────────────────────────────────┘
```

### 混合训练：新数据 + 旧数据防止灾难性遗忘

```python
class FlyWheelTrainer:
    """管理飞轮的训练循环"""
    
    def __init__(self, base_model, memory_ratio: float = 0.3):
        self.base_model = base_model
        self.memory_ratio = memory_ratio  # 旧数据占比（防止遗忘）
        self.all_historical_data = []     # 所有历史高质量数据
    
    def prepare_training_data(self, 
                               new_high_quality: list[AgentTrajectory],
                               max_samples: int = 20000) -> list[dict]:
        """
        混合新旧数据：
        - 70% 新数据（当前迭代收集的）
        - 30% 旧数据（防止灾难性遗忘）
        """
        self.all_historical_data.extend(new_high_quality)
        
        # 新数据
        new_samples = [self._to_training_format(t) for t in new_high_quality]
        
        # 历史数据（按质量分数加权采样）
        n_historical = int(max_samples * self.memory_ratio)
        historical_sample = self._weighted_sample(
            self.all_historical_data, 
            n=min(n_historical, len(self.all_historical_data)),
            weight_key="auto_quality_score"
        )
        historical_formatted = [self._to_training_format(t) for t in historical_sample]
        
        # 合并 + 打散
        import random
        combined = new_samples + historical_formatted
        random.shuffle(combined)
        
        print(f"训练数据：{len(new_samples)} 新 + {len(historical_formatted)} 旧 = {len(combined)} 总")
        return combined[:max_samples]
    
    def _weighted_sample(self, data, n, weight_key):
        """按质量分数加权采样（质量高的被选中概率更大）"""
        import random
        weights = [getattr(d, weight_key, 0.5) for d in data]
        return random.choices(data, weights=weights, k=min(n, len(data)))
    
    def run_iteration(self, 
                       new_trajectories: list[AgentTrajectory],
                       iteration: int) -> str:
        """运行一次飞轮迭代"""
        print(f"\n{'='*50}")
        print(f"飞轮第 {iteration} 轮迭代")
        print(f"{'='*50}")
        
        # 1. 过滤
        filter = TrajectoryFilter()
        high_quality = filter.filter_batch(new_trajectories, min_score=0.72)
        print(f"✅ 高质量样本：{len(high_quality)} 条")
        
        # 2. 提取失败案例（用于 DPO）
        low_quality = [t for t in new_trajectories 
                       if t.auto_quality_score < 0.4]
        print(f"📊 低质量样本（DPO 负样本）：{len(low_quality)} 条")
        
        # 3. 准备训练数据
        training_data = self.prepare_training_data(high_quality)
        
        # 4. SFT 微调
        output_path = f"./agent-v{iteration}/sft"
        self._run_sft(training_data, output_path)
        
        # 5. DPO（如果有足够的偏好对）
        if len(low_quality) >= 100:
            self._run_dpo(high_quality, low_quality, f"./agent-v{iteration}/dpo")
        
        # 6. 评估新版本
        eval_results = self._evaluate_on_benchmark()
        print(f"📈 评估结果（迭代 {iteration}）：{eval_results}")
        
        return f"./agent-v{iteration}"
    
    def _run_sft(self, data: list[dict], output_path: str):
        """执行 SFT 训练"""
        print(f"开始 SFT 训练（{len(data)} 条样本）...")
        # ... 实际调用 Trainer
    
    def _run_dpo(self, chosen_trajs, rejected_trajs, output_path: str):
        """执行 DPO 训练"""
        print("开始 DPO 对比学习...")
        # ... 实际调用 DPOTrainer
    
    def _evaluate_on_benchmark(self) -> dict:
        """在标准基准上评估"""
        return {
            "tool_accuracy": 0.0,  # 占位，实际运行评估
            "task_completion": 0.0,
            "format_validity": 0.0,
        }
```

---

## 飞轮的三个关键加速因素

### 1. 任务难度课程（Curriculum）

```python
class CurriculumManager:
    """
    从简单任务开始，逐步增加难度。
    简单任务 → 基础工具使用 → 多步推理 → 错误恢复 → 复杂长链
    """
    
    DIFFICULTY_LEVELS = {
        1: {"max_tool_calls": 1, "max_steps": 2},   # 单工具调用
        2: {"max_tool_calls": 3, "max_steps": 5},   # 顺序多工具
        3: {"max_tool_calls": 5, "max_steps": 8},   # 条件分支
        4: {"max_tool_calls": 8, "max_steps": 12},  # 复杂多步推理
        5: {"max_tool_calls": None, "max_steps": None},  # 无限制
    }
    
    def get_difficulty_for_iteration(self, iteration: int) -> int:
        """随迭代逐步提升难度"""
        return min(5, 1 + iteration // 3)  # 每3轮提升一级
    
    def filter_by_difficulty(self, 
                              trajectories: list[AgentTrajectory],
                              level: int) -> list[AgentTrajectory]:
        """只保留目标难度级别的轨迹"""
        config = self.DIFFICULTY_LEVELS[level]
        result = []
        for traj in trajectories:
            if (config["max_tool_calls"] is None or 
                    traj.total_tool_calls <= config["max_tool_calls"]):
                result.append(traj)
        return result
```

### 2. 探索性采样（增加多样性）

```python
class ExploratoryAgentRunner:
    """
    在线上 Agent 中引入探索：
    用更高的 temperature 运行一部分请求，
    发现"不常走的路径"，丰富训练数据多样性。
    """
    
    def __init__(self, base_agent, exploration_rate: float = 0.1):
        self.base_agent = base_agent
        self.exploration_rate = exploration_rate
    
    async def run(self, user_request: str) -> tuple[str, AgentTrajectory]:
        """运行 Agent，部分请求使用探索性配置"""
        import random
        
        is_exploratory = random.random() < self.exploration_rate
        
        if is_exploratory:
            # 高 temperature 激励探索
            config = {"temperature": 0.8, "tag": "exploratory"}
        else:
            # 正常生产配置
            config = {"temperature": 0.2, "tag": "production"}
        
        response, trajectory = await self.base_agent.run(
            user_request, **config
        )
        trajectory.metadata = config
        
        return response, trajectory
```

### 3. 合成数据扩充（覆盖盲区）

```python
class BlindSpotFixer:
    """
    分析现有训练数据的盲区，有针对性地合成补充。
    
    例如：发现模型在"工具调用失败后的处理"上准确率只有 40%，
    就专门合成 500 条这类场景的轨迹。
    """
    
    def identify_weaknesses(self, 
                             eval_results: dict,
                             threshold: float = 0.6) -> list[str]:
        """识别表现低于阈值的技能维度"""
        weak_skills = []
        for skill, score in eval_results.items():
            if score < threshold:
                weak_skills.append(skill)
        return weak_skills
    
    async def synthesize_for_weakness(self, 
                                       skill: str,
                                       n: int = 500) -> list[dict]:
        """针对薄弱技能合成专项训练数据"""
        
        skill_to_scenario = {
            "error_recovery": "工具调用返回错误，Agent 需要正确处理",
            "tool_selection": "多个工具可选，需要正确判断使用哪个",
            "format_validity": "需要输出严格的 JSON 格式工具调用",
            "multi_step_planning": "需要多步工具调用才能完成的复杂任务",
        }
        
        scenario = skill_to_scenario.get(skill, skill)
        
        # 批量合成
        # ... 调用合成管道
        print(f"合成 {n} 条 '{skill}' 场景训练数据...")
        return []  # 实际实现
```

---

## 飞轮的效果：现实案例参考

| 团队 | 方法 | 迭代次数 | 效果 |
|------|------|---------|------|
| **DeepSeek** | GRPO + 自生成数学轨迹 | ~10 轮 | 数学推理从 GPT-4 级别追上 o1 |
| **Reflection-70B** | Self-reflection 自我批评 | ~5 轮 | Llama 70B 超越 GPT-4 (有争议) |
| **STaR / V-STaR** | 用正确推理链自举 | 5 轮 | 数学准确率 +40% |
| **AgentTuning** | 多任务 Agent 轨迹微调 | 1 轮 | 通用 Agent 能力 +30% |

> 📌 核心规律：**数据飞轮前 3 轮改善最快**（模型从"不会用工具"到"会用工具"的跨越），后续每轮收益递减，需要更精细的数据工程。

---

## 实战检查清单

开始构建你的 Agentic 数据飞轮之前，确认以下条件：

```
基础条件：
□ Agent 线上系统已稳定运行（每天 > 100 次调用）
□ 轨迹记录系统已部署（收集 system/user/tool/assistant 完整上下文）
□ 用户反馈渠道已接入（点赞/踩、满意度评分）

数据管道：
□ 脱敏流水线已就绪（GDPR/PIPL 合规）
□ 质量评分函数已实现并在 100 条样本上验证
□ 存储系统能支撑每天增量（建议用对象存储 + Parquet 格式）

训练条件：
□ 有 GPU 资源（至少 A100/H100 × 1，建议 × 4）
□ 训练代码已验证（本地小规模跑通）
□ 评估基准已定义（工具准确率、任务完成率等）

节奏规划：
□ 明确迭代周期（建议 2-4 周一次）
□ 有版本对比的 A/B 测试方案
□ 明确何时"停止迭代"的条件（边际收益低于阈值）
```

---

## 本节小结

Agentic 数据飞轮的本质是：**用 Agent 自身的运行数据来训练更强的 Agent，形成自我强化循环**。

```
核心环节：

收集 → 过滤 → 标注 → 训练 → 部署 → 回到收集

每一环的关键：
┌─────────────────────────────────────────────────────────────┐
│ 收集：记录完整轨迹，不只是输入输出，还要有工具调用细节        │
│ 过滤：双维度（结果 × 过程），质量分 > 0.7 才入训练集         │
│ 标注：可验证任务用规则奖励；不可验证用 LLM-as-Judge           │
│ 训练：新数据 70% + 历史数据 30%（防止灾难性遗忘）            │
│ 部署：保留 10% 探索流量（发现新场景）                        │
└─────────────────────────────────────────────────────────────┘
```

飞轮启动需要初始模型和初始数据，但一旦转起来，**数据质量和模型能力会互相拉动提升**。这也是为什么"先发优势"在 Agent 领域如此重要——早一步启动飞轮，就早一步积累别人赶不上的数据优势。

> 🔗 **与前序章节的关系**：  
> - **第 3.8 节**：训练数据的质量工程（工程基础）  
> - **第 11.2 节**：SFT + LoRA 训练方法（训练技术）  
> - **第 11.3-11.5 节**：PPO/DPO/GRPO 算法（奖励信号）  
> - **第 11.8 节**：Agent 专项数据构建（数据来源）  
> - **本节（11.9）**：将以上所有内容串成一个持续迭代的系统

---

## 参考文献

1. Zelikman et al. "STaR: Bootstrapping Reasoning With Reasoning." NeurIPS 2022.
2. Singh et al. "Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models (V-STaR)." 2024.
3. Zeng et al. "AgentTuning: Enabling Generalized Agent Abilities for LLMs." 2023.
4. Chen et al. "Self-play Fine-tuning Converts Weak Language Models to Strong Language Models (SPIN)." ICML 2024.
5. Guo et al. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." DeepSeek 2025.
6. Mitra et al. "AgentInstruct: Toward Generative Teaching with Agentic Flows." Microsoft Research 2024.
