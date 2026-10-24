# 19.7 Guardrails运行时防护

Guardrails是围绕输入、提议、执行与结果设置的检查。先写清每个检查的输入、失败状态与执行位置，再考虑选择框架。本节直接运行本地机制，不把未经安装的第三方配置称为完整防护系统。

![信号与强制执行分别进入运行时](../svg/chapter_21_security_07_guardrails_chain.svg)

## 从顺序合同理解运行时

`runtime.py`的`execute`按以下顺序执行：

1. 先检查总尝试预算，再扣除本次尝试；无效输入也计数。
2. 有界JSON解析，拒绝重复键、非有限常量、额外字段和不符类型。
3. 按封闭工具表与具体参数Schema校验；身份和审批不在模型Schema内。
4. 检查当前工具许可、资源范围和文档归属。
5. 写入动作先查已存在的内容绑定收据；新动作需要有效审批。
6. 立即操作前再次鉴权；后端仍检查其负责的资源条件。
7. 返回有界夹具结果，写入只含元数据的审计条目。

这里的“失败时拒绝”具体指格式/权限/审批检查失败不执行动作，检测器或工具异常也不转换为成功。它不表示程序能够识别所有未知攻击。与源码对应的状态为`succeeded/rejected/failed`，固定原因码帮助区分重试、修参、补授权与停止。

## 同一预算涵盖坏输入、拒绝和错误

```python
from reference_code.security_boundary.regression import new_lab, NOTICE, READ

runtime = new_lab(max_attempts=2)
assert runtime.execute('{')['status'] == 'rejected'
assert runtime.execute(NOTICE)['code'] == 'approval_required'
assert runtime.execute(READ)['code'] == 'budget_exhausted'
assert runtime.backend.read_calls == 0
assert runtime.backend.publish_calls == 0
assert runtime.attempts == 2
assert len(runtime.audit) == 2
print('Invalid and denied proposals consume the same total attempt budget.')
```

预算耗尽后不继续扩充审计列表，因此该内存记录受最多100次尝试约束。单次提议最多4096 UTF-8字节，公告文字最多600字节，审批/便签各最多100条。它们是教学上限，不是经过负载测量的生产默认值。内存后端没有长任务或外部响应；真实工具还要独立落实超时、取消、结果体积及每租户配额。

## 错误不能泄露参数或变成成功

下面故意让实际读取失败。错误类型的细节只适合受控内部诊断，面向模型的结果与本练习审计均不复制原始异常：

```python
from unittest.mock import patch
import json
from reference_code.security_boundary.regression import new_lab, READ

runtime = new_lab(max_attempts=2)
with patch.object(runtime.backend, 'read', side_effect=RuntimeError('SYNTHETIC_SECRET')):
    result = runtime.execute(READ)
assert result == {'status':'failed', 'code':'tool_error'}
assert runtime.attempts == 1
serialized = json.dumps([result, runtime.audit])
assert 'SYNTHETIC_SECRET' not in serialized
assert 'arguments' not in serialized
print(result)
```

`KeyboardInterrupt`等进程控制信号不被转换为普通工具错误。异常前若远程动作可能已完成，应查询幂等记录/对账，而不是盲目重发；本地已提交后丢失返回的回归见测试。使用fallback时也不能把“替代文字”标为原动作已成功。

## 把软信号和硬条件放在各自位置

| 检查 | 实际能执行的条件 | 还需要什么 |
|---|---|---|
| 类型/大小/工具表 | 当前解析器与注册表的闭合合同 | 保持前后端Schema一致 |
| 身份/资源策略 | 可信主体、明确能力和资源范围 | 真正鉴权、后端执行与并发条件 |
| 内容检测器/模型判卷 | 已配置规则或分类器的判定 | 校准误报/漏报；异常不得当通过 |
| 审批 | 具体内容、主体、期限绑定 | 审批者身份与可信展示 |
| 隔离服务 | 实际文件/网络/资源约束 | 独立部署与逃逸/清理验收 |

正则规则可提供窄范围信号，模型检查器可处理更复杂文本，但都不能直接授予权限。仓库`reference-agent`中的`InjectionGuard`因此明确命名为有界词句筛查，`check_input`返回只表示没有命中这些规则，不表示语义安全；其测试同时保留误报与漏报。

选用第三方Guardrails框架时，固定版本与依赖，核验它究竟包住输入、输出、工具调用还是服务后端。逐项实测配置不生效、解析失败、超时、取消与绕过入口。本文没有安装或比较NeMo、Guardrails AI等产品，不给出“最成熟”排名、通用延迟数字或防绕过保证。

练习：故意漏扣一次拒绝请求的预算，确认独立测试能发现它；再模拟工具已写入但返回丢失，验证同内容重试没有第二次副作用。提交状态与本地实际效果，不只提交日志截图。

*下一节：[19.8 红队测试方法论](./08_red_teaming.md)*
