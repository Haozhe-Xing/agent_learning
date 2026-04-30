# 9.3 技能学习与获取

上一节介绍了如何**手动定义**技能。但更令人兴奋的问题是：**Agent 能否自主学习新技能？**

本节探讨 Agent 技能学习的三种范式：从经验中学习、从工具中创造、以及通过蒸馏获取。

## 范式一：从经验中学习——Voyager 模式

### 核心思想

Voyager（NVIDIA, 2023）是 Agent 自主技能学习的里程碑之作。它在 Minecraft 游戏中展示了一个惊人的能力循环：

> **探索** → **尝试** → **成功** → **保存为技能** → **复用技能** → **探索更复杂的任务**

就像人类学骑自行车一样——先摔几次，掌握平衡后就变成了"肌肉记忆"（技能），之后就能自动执行，不需要每次都重新学习。

### Voyager 的三大组件

![Voyager 架构](../svg/chapter_skill_03_voyager_arch.svg)

### 技能学习的完整流程

```python
# Voyager 技能学习流程（简化版 Python 伪代码）

class VoyagerAgent:
    def __init__(self):
        self.skill_library = SkillLibrary()
        self.curriculum = AutoCurriculum()
        self.code_generator = CodeGenerator()  # GPT-4
        self.critic = TaskCritic()  # GPT-4
    
    def learning_loop(self):
        """核心学习循环"""
        while True:
            # 1. 课程系统提出下一个任务
            task = self.curriculum.propose_next_task(
                completed_tasks=self.completed_tasks,
                current_state=self.get_env_state()
            )
            print(f"📋 新任务: {task}")
            
            # 2. 从技能库检索相关技能
            relevant_skills = self.skill_library.retrieve(task, top_k=5)
            print(f"📚 找到 {len(relevant_skills)} 个相关技能")
            
            # 3. 生成代码（结合已有技能）
            success = False
            for attempt in range(4):  # 最多尝试 4 次
                code = self.code_generator.generate(
                    task=task,
                    relevant_skills=relevant_skills,
                    env_state=self.get_env_state(),
                    previous_errors=self.errors if attempt > 0 else None
                )
                
                # 4. 执行代码
                result = self.execute(code)
                
                # 5. 自我验证
                success, critique = self.critic.check(
                    task=task,
                    result=result,
                    env_state=self.get_env_state()
                )
                
                if success:
                    break
                else:
                    self.errors = critique
                    print(f"  ❌ 尝试 {attempt + 1} 失败: {critique}")
            
            # 6. 如果成功，保存为新技能
            if success:
                skill_name = self.extract_skill_name(task)
                description = self.generate_description(task, code)
                self.skill_library.add(skill_name, code, description)
                self.curriculum.mark_completed(task)
                print(f"  ✅ 新技能习得: {skill_name}")
            else:
                print(f"  ⚠️ 任务失败，跳过")
```

### Voyager 的关键洞察

1. **技能是可组合的**：简单技能组合成复杂技能
   ```
   "挖木头" + "制作木板" + "制作工作台" 
     → "建造基础设施"（组合技能）
   ```

2. **技能是可迁移的**：在不同场景中复用
   ```
   "挖木头" 技能可以迁移到 "挖石头"（相似的操作模式）
   ```

3. **技能库持续增长**：新技能建立在旧技能之上
   ```
   第 1 小时：10 个基础技能
   第 5 小时：50+ 个技能（含组合技能）
   第 20 小时：100+ 个技能（覆盖游戏的大部分行为）
   ```

### 从 Voyager 到通用 Agent

Voyager 的思想可以推广到通用 Agent 开发中：

```python
# 通用 Agent 的技能学习系统
class LearningAgent:
    """能从经验中学习技能的通用 Agent"""
    
    def __init__(self):
        self.skill_library = SkillLibrary()
    
    def execute_task(self, task: str):
        """执行任务，并从成功经验中提取技能"""
        
        # 检索已有技能
        skills = self.skill_library.retrieve(task)
        
        # 使用 LLM + 已有技能完成任务
        result = self.llm_execute(task, skills)
        
        # 如果任务成功，评估是否值得保存为新技能
        if result.success and self._is_reusable(task, result):
            self._save_as_skill(task, result)
    
    def _is_reusable(self, task: str, result) -> bool:
        """判断经验是否值得保存为技能"""
        # 条件：
        # 1. 任务成功完成
        # 2. 解决方案包含多个步骤（不是简单的单步操作）
        # 3. 类似的任务可能再次出现
        # 4. 技能库中没有高度相似的技能
        return (result.success 
                and result.steps > 2 
                and not self.skill_library.has_similar(task))
    
    def _save_as_skill(self, task: str, result):
        """将成功经验保存为技能"""
        skill = {
            "name": self._generate_skill_name(task),
            "description": self._generate_description(task, result),
            "procedure": result.steps,  # 执行步骤
            "tools_used": result.tools,  # 使用的工具
            "preconditions": self._extract_preconditions(task),
            "success_criteria": self._extract_criteria(task, result)
        }
        self.skill_library.add(skill)
```

---

## 范式二：从工具中创造——CRAFT 模式

### 核心思想

CRAFT（ICLR 2024）提出了一个不同于 Voyager 的思路：**不仅使用现有工具，还能创造新工具**。

```
传统方法：
  问题 → 用已有工具解决 → 如果没有合适工具就卡住

CRAFT 方法：
  问题 → 先创造解决这类问题的专用工具 → 用新工具解决
  如果下次遇到类似问题 → 直接检索已创造的工具
```

### CRAFT 的工作流程

![CRAFT 工作流程](../svg/chapter_skill_03_craft_flow.svg)

```
1. 创建阶段（Create）
   问题："计算给定数组中所有质数的和"
     ↓
   LLM 创建专用工具函数：
   def sum_of_primes(arr):
       def is_prime(n):
           if n < 2: return False
           for i in range(2, int(n**0.5)+1):
               if n % i == 0: return False
           return True
       return sum(x for x in arr if is_prime(x))
   
2. 验证阶段（Verify）
   用测试用例验证工具的正确性
   sum_of_primes([1,2,3,4,5]) == 10  ✅
   
3. 存储阶段（Store）
   将验证通过的工具保存到工具库
   
4. 检索阶段（Retrieve）
   下次遇到类似问题时，检索已有工具
   "计算数组中所有偶数的和" → 检索到 sum_of_primes
   → 参考其结构创建 sum_of_evens
```

### CRAFT 的 Python 实现

```python
class CRAFTSystem:
    """CRAFT：创建和检索专用工具集"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tool_library = {}  # 工具名 → (代码, 描述, 测试)
        self.embeddings = {}
    
    def solve(self, problem: str) -> str:
        """解决问题：先检索/创建工具，再用工具解决"""
        
        # 1. 检索已有工具
        relevant_tools = self._retrieve_tools(problem)
        
        if relevant_tools:
            # 用已有工具解决
            return self._solve_with_tools(problem, relevant_tools)
        
        # 2. 没有合适工具 → 创建新工具
        new_tool = self._create_tool(problem)
        
        # 3. 验证新工具
        if self._verify_tool(new_tool):
            self._store_tool(new_tool)
            return self._solve_with_tools(problem, [new_tool])
        else:
            # 工具验证失败，回退到直接解决
            return self._direct_solve(problem)
    
    def _create_tool(self, problem: str) -> dict:
        """让 LLM 创建专用工具"""
        prompt = f"""分析以下问题，创建一个可复用的 Python 工具函数来解决这类问题。

问题：{problem}

要求：
1. 函数应该通用化，能解决这一类问题（不仅仅是这一个具体实例）
2. 提供清晰的函数签名和文档字符串
3. 提供至少 3 个测试用例

请返回：
- function_name: 函数名
- code: 完整的 Python 代码
- description: 功能描述
- test_cases: 测试用例列表
"""
        result = self.llm.generate(prompt)
        return parse_tool_response(result)
    
    def _verify_tool(self, tool: dict) -> bool:
        """用测试用例验证工具"""
        try:
            exec(tool["code"])
            func = eval(tool["function_name"])
            for test in tool["test_cases"]:
                assert func(*test["input"]) == test["expected"]
            return True
        except Exception as e:
            print(f"工具验证失败: {e}")
            return False
```

### CRAFT vs Voyager

| 维度 | Voyager | CRAFT |
|------|---------|-------|
| **学习场景** | 具身环境（Minecraft） | 通用问题求解 |
| **技能类型** | 操作序列（行为技能） | 工具函数（计算技能） |
| **学习信号** | 环境反馈（成功/失败） | 测试用例验证 |
| **组合方式** | 序列组合 | 函数调用组合 |
| **关键创新** | 自动课程 + 技能库 | 创建 + 检索 + 验证 |

---

## 范式三：技能蒸馏——大模型到小模型

### 核心思想

大模型（如 GPT-4、Claude）天然具备丰富的"隐式技能"，但部署成本高。**技能蒸馏**将大模型的技能转移到更小、更高效的模型中：

```
技能蒸馏流程：
  1. 让大模型（GPT-4）执行大量任务
  2. 记录每次执行的"思考过程"和"行动序列"
  3. 用这些数据微调小模型（如 Qwen-7B）
  4. 小模型获得了特定领域的技能
  
类比：
  大模型 = 经验丰富的师傅
  执行记录 = 师傅的操作录像
  微调训练 = 学徒看录像学习
  小模型 = 学会了特定技艺的学徒
```

### 实际应用

```python
# 技能蒸馏的数据收集过程
def collect_skill_demonstrations(
    teacher_model,  # GPT-4
    tasks: list,
    skill_name: str
):
    """收集大模型的技能演示数据"""
    demonstrations = []
    
    for task in tasks:
        # 让大模型执行任务，记录完整的推理和行动
        result = teacher_model.execute(
            task=task,
            return_reasoning=True,  # 返回思考过程
            return_actions=True     # 返回行动序列
        )
        
        if result.success:
            demonstrations.append({
                "task": task,
                "reasoning": result.reasoning,
                "actions": result.actions,
                "output": result.output,
                "skill": skill_name
            })
    
    return demonstrations

# 用收集的数据微调小模型
def distill_skill(
    student_model,  # Qwen-7B
    demonstrations: list
):
    """将技能蒸馏到小模型"""
    training_data = []
    for demo in demonstrations:
        training_data.append({
            "messages": [
                {"role": "system", "content": f"你是一个具备{demo['skill']}技能的 Agent。"},
                {"role": "user", "content": demo["task"]},
                {"role": "assistant", "content": f"<think>{demo['reasoning']}</think>\n{demo['output']}"}
            ]
        })
    
    # 微调小模型
    student_model.fine_tune(training_data)
```

### DeepSeek-R1 蒸馏的启示

DeepSeek-R1（2025）的蒸馏实验证明了技能蒸馏的巨大潜力：

![DeepSeek-R1 蒸馏链](../svg/chapter_skill_03_deepseek_distill.svg)

---

## 技能进化：从简单到复杂

不管是哪种学习范式，技能都遵循从简单到复杂的进化路径：

```
Level 1：原子技能（Atomic Skills）
  单个工具调用或简单操作
  例：读取文件、发送 HTTP 请求、格式化文本

Level 2：组合技能（Composite Skills）
  多个原子技能的序列组合
  例：数据加载 → 清洗 → 分析（3个原子技能组合）

Level 3：策略技能（Strategic Skills）
  包含条件判断和分支的复杂技能
  例：根据数据类型自动选择分析方法

Level 4：元技能（Meta Skills）
  管理和组合其他技能的高级技能
  例：自动分解任务 → 选择合适的技能 → 编排执行顺序

Level 5：自进化技能（Self-Evolving Skills）
  能够自我改进和创造新技能的技能
  例：Voyager 的学习循环、CRAFT 的工具创造
```

## 本节小结

| 学习范式 | 代表 | 核心机制 | 适用场景 |
|---------|------|---------|---------|
| **经验学习** | Voyager | 尝试 → 成功 → 保存 | 探索性任务、具身智能 |
| **工具创造** | CRAFT | 分析 → 创建 → 验证 → 存储 | 问题求解、算法任务 |
| **技能蒸馏** | DeepSeek-R1 | 大模型演示 → 数据收集 → 微调 | 成本优化、边缘部署 |

> 💡 **核心观点**：Agent 技能学习的终极目标是**终身学习（Lifelong Learning）**——Agent 在持续与环境交互的过程中不断积累新技能，技能库持续增长。Voyager 展示了这种可能性：从零开始，通过数小时的自主探索，学会了 100+ 个技能。未来的 Agent 将像人类一样，在工作中不断学习和成长。

---

*下一节：[9.4 技能发现与注册](./04_skill_discovery.md)*
