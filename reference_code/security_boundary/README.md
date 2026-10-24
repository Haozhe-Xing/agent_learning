# 本地授权边界练习 / Local authorization boundary lab

第19章独立参考代码，Python 3.11+、仅标准库。在仓库根目录运行：

```bash
python -m reference_code.security_boundary
python -m unittest examples.integration_tests.test_security_chapter -v
```

`runtime.py`实现封闭工具/参数校验、宿主身份与资源许可、审批摘要/期限绑定、执行前复查、本地收据幂等和总尝试预算。`privacy.py`实现允许字段投影、明确同意、主体分区的内存便签及删除。`regression.py`的11个冻结用例独立检查返回值和实际本地状态：3个正常任务、8个越权提议。CLI报告这些固定提议的结果，不调用模型，不代表真实模型的注入防御成功率。

Only the host may construct/change policy, issue approvals and access the backend. A model receives proposal schemas, not these Python objects. `publish_notice` adds a fictional in-memory notice and receipt; it never sends a message. Failed parsing, denial and handler errors all consume attempts. Audit entries omit raw arguments, documents, proof tokens and exception messages.

身份是宿主给定的夹具，不含登录或审批界面。单进程同步内存状态不证明并发原子性、重启恢复、远程恰好一次或磁盘数据保护。审批容量100、便签容量100、单次提议4096 UTF-8字节、文字600字节，运行最多100次尝试；实际外部服务还需要独立结果大小、超时、取消、租户配额及存储生命周期。不要把真实密钥放进本练习。

There is no arbitrary code executor, file/browser/network tool or OS sandbox. The lexical signal deliberately has false positives and false negatives; `no_match` never grants authority. No general PII detector or encryption is implemented. See the chapter for the separate isolation and key-management contract.

目录可整体复制为`security_boundary/`，从其父目录运行`python -m security_boundary`，不依赖本仓库其余实现。课程主线仍是`reference_code/evidence_assistant/`，本目录是选做机制练习。
