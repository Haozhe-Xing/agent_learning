# 5.6 Plan-and-Execute 与 Test-time Compute Scaling

本节保留已有的有界执行队列，并把它与5.3的依赖检查、5.4的有限修订组合。计划执行和任务验收分别记录；额外计算是否值得，取决于能否在同一任务合同和总预算下改善结果，不是模式名称的优胜排序。

## 先看执行完成不等于交付通过

共享的 `reference_code/evidence_assistant/planning.py` 中，`completed`表示队列已执行且工具没有抛出异常。它不验证产物内容。原有入口 `examples/chapter05/plan_and_execute.py`及队列回归仍保留；下面直接展示两种判定的差别：

```python
from reference_code.evidence_assistant.planning import PlanAndExecuteAgent, Step
from reference_code.planning_patterns import FACTS, check_brief

def planner(task):
    return [Step("proposal")]

def proposal():
    return {"budget": {"value": "300 units", "source": "budget-v1#release"}}

execution = PlanAndExecuteAgent(planner, {"proposal": proposal}, max_steps=1).run("brief")
assert execution["status"] == "completed"  # the tool returned, not task acceptance
candidate = execution["trace"][0]["result"]
assert check_brief(candidate, FACTS)["status"] == "failed"
```

`planning_patterns.run_plan`复用该队列，而非复制另一份同名Agent；外层增加计划校验、固定观察与独立检查，再给出 `status/accepted`。从仓库根运行：

```bash
python -m reference_code.planning_patterns plan
python -m reference_code.planning_patterns revise
python -m reference_code.planning_patterns stubborn
python -m unittest examples.integration_tests.test_planning_chapter -v
```

前三个命令分别演示直接通过、修订后通过、持续缺项耗尽。最后一种退出码为1；输出保留候选、来源、缺失/不受支持项、公开动作轨迹和实际预算。

## 有界队列与替换计划的契约

| 计量 | 本实现的含义 | 不代表什么 |
|---|---|---|
| `max_steps` | 全任务工具尝试，含失败与新计划中的动作；整数1..30 | 模型token或墙钟秒数 |
| `max_replans` | 接受要点失败后的替换计划机会；整数0..3 | 已有产物可不经检查复用 |
| 计划大小 | 每版最多12节点，先校验DAG和目标映射 | 自动HTN规划、并行执行 |
| 资料/产物/反馈大小 | 8192/8192/4096序列化字节 | 真实模型请求的完整token计量 |

失败后替换pending队列，不是在 `for enumerate(plan)`中重绑一个列表。当前实现保守清空旧产物再执行新依赖，失败动作仍占用总预算。空替代计划、无效反思、读取错误或权限拒绝都不产生成功；不会通过更换工具绕过许可。

```python
from reference_code.planning_patterns import OmitRollbackOnce, run_plan

result = run_plan(writer=OmitRollbackOnce(), max_steps=3, max_replans=1)
assert result["status"] == "budget_exhausted"
assert result["accepted"] is False
assert result["budgets"]["steps_used"] == 3
assert result["verification"]["status"] == "not_started"

try:
    run_plan(max_replans=True)
except ValueError:
    print("a Boolean is not a replanning budget")
else:
    raise AssertionError("invalid budget was accepted")
```

三次工具额度中已经有第一次读取、失败写作与重读，没有额度形成新候选，因此不能沿用旧核验或自称完成。整数预算在执行前检查；`True`、浮点数和负数不是可接受的计数。

## ReAct与先规划后执行如何选择

已知输入依赖时，先校验完整小计划容易发现漏目标；需要新观察才能选择动作时，可用5.2的逐步结构决策。两种方式都需要状态、权限、预算与独立验收。ReAct也可以维护长期计划，先规划也可能按观察改动，二者不是互斥产品栈。

这个练习的ReAct修订可以复用已经读取的观察，队列替换则刻意重读，形成可检查的成本差异；这不是质量或速度的同题模型实验。是否增加状态图取决于恢复与协作需求，见[第11章](../chapter_langgraph/README.md)；编译图不会自动带来磁盘持久化或一次性副作用。

## 推理时计算：有限预算内比较，而非越多越好

[Snell等，v1](https://arxiv.org/abs/2408.03314v1)研究推理时计算分配，并在给定模型、难度与计算匹配条件下比较策略。不能推广为“任何任务多推理几步都比增大模型有效”。反复生成、候选搜索、验证与重试均有成本，错误反馈也可能使结果退化。

| 策略 | 潜在作用 | 必须单独测量 |
|---|---|---|
| 有限候选与独立检查 | 增加找到合格产物的机会 | 生成/评分总次数、相关错误、未通过比例 |
| 反馈修订 | 根据明确缺项再提议 | 是否补齐、是否引入新错、总预算 |
| 搜索树/MCTS | 在有明确定义的状态、动作与奖励下搜索 | 分支/深度/节点/时间上限、奖励偏差与独立最终验收 |

本节不提供MCTS、模型难度分类器或推理token控制接口，不把缺方法的骨架当完整搜索器，也不把访问最多的路径称作最优解。真实服务中的推理参数依模型和SDK而异，不能统一设置“16000+ token”或把所有历史都塞入上下文。

练习固定同一资料和独立要求，对比直接写作、一次反馈修订及不修复的结果；记录实际尝试、缺项、总输入和停止原因。若要比较真实模型，先补充问题集、提供者用量与耗时测量，不能从固定替身的调用次数推导价格或性能排名。

*下一节：[5.7 论文解读：规划与推理前沿研究](./06_paper_readings.md)*
