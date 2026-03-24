# 敏感数据保护

> **本节目标**：学会识别和保护 Agent 交互中的敏感数据，防止数据泄露。

![Agent 敏感数据保护流程](../svg/chapter_security_04_data_protection.svg)

---

## Agent 面临的数据风险

Agent 在与用户交互和执行任务的过程中，可能会接触到各种敏感数据：

| 数据类型 | 示例 | 风险 |
|---------|------|------|
| 个人身份信息（PII） | 姓名、身份证号、手机号 | 隐私泄露 |
| 认证凭据 | API Key、密码、Token | 账号被盗 |
| 金融数据 | 银行卡号、交易记录 | 财务损失 |
| 业务机密 | 内部文档、代码、策略 | 商业损失 |

---

## PII 检测与脱敏

```python
import re
from dataclasses import dataclass

@dataclass
class PIIEntity:
    """个人身份信息实体"""
    type: str       # 类型
    value: str      # 原始值
    masked: str     # 脱敏后的值
    start: int      # 在文本中的起始位置
    end: int        # 结束位置


class PIIDetector:
    """PII 检测器"""
    
    PATTERNS = {
        "phone": {
            "regex": r"1[3-9]\d{9}",
            "mask": lambda m: m[:3] + "****" + m[-4:]
        },
        "id_card": {
            "regex": r"\d{17}[\dXx]",
            "mask": lambda m: m[:6] + "********" + m[-4:]
        },
        "email": {
            "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "mask": lambda m: m[0] + "***@" + m.split("@")[1]
        },
        "bank_card": {
            "regex": r"\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}",
            "mask": lambda m: re.sub(r'\d', '*', m[:-4]) + m[-4:]
        },
        "api_key": {
            "regex": r"(sk|pk|api)[_-][a-zA-Z0-9]{20,}",
            "mask": lambda m: m[:6] + "*" * (len(m) - 10) + m[-4:]
        }
    }
    
    def detect(self, text: str) -> list[PIIEntity]:
        """检测文本中的 PII"""
        entities = []
        
        for pii_type, config in self.PATTERNS.items():
            for match in re.finditer(config["regex"], text):
                entities.append(PIIEntity(
                    type=pii_type,
                    value=match.group(),
                    masked=config["mask"](match.group()),
                    start=match.start(),
                    end=match.end()
                ))
        
        return entities
    
    def mask(self, text: str) -> tuple[str, list[PIIEntity]]:
        """检测并脱敏文本中的 PII"""
        entities = self.detect(text)
        
        # 从后向前替换，避免位置偏移
        masked_text = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            masked_text = (
                masked_text[:entity.start] + 
                entity.masked + 
                masked_text[entity.end:]
            )
        
        return masked_text, entities


# 使用示例
detector = PIIDetector()

text = "用户张三的手机号是13812345678，邮箱是zhangsan@example.com"
masked, entities = detector.mask(text)

print(masked)
# 用户张三的手机号是138****5678，邮箱是z***@example.com

print(f"发现 {len(entities)} 个 PII：")
for e in entities:
    print(f"  [{e.type}] {e.value} → {e.masked}")
```

---

## Agent 对话数据保护

```python
class SecureConversationManager:
    """安全的对话管理器 —— 在存储前脱敏"""
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        self.conversations = {}  # session_id -> messages
    
    def add_message(self, session_id: str, role: str, content: str):
        """添加消息（自动脱敏后存储）"""
        
        # 脱敏处理
        masked_content, entities = self.pii_detector.mask(content)
        
        if entities:
            print(f"⚠️ 检测到 {len(entities)} 个敏感信息，已脱敏")
        
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": masked_content,  # 存储脱敏后的内容
            "has_pii": len(entities) > 0,
            "pii_types": list(set(e.type for e in entities))
        })
    
    def get_history(self, session_id: str) -> list[dict]:
        """获取对话历史（已脱敏）"""
        return self.conversations.get(session_id, [])
```

---

## 数据最小化原则

只收集和处理完成任务所必需的数据：

```python
class DataMinimizer:
    """数据最小化处理器"""
    
    @staticmethod
    def minimize_for_llm(data: dict, task: str) -> dict:
        """根据任务类型，只向 LLM 传递必要的数据字段"""
        
        # 不同任务需要的字段
        task_fields = {
            "order_query": ["order_id", "status", "create_time"],
            "product_recommend": ["preferences", "budget"],
            "complaint": ["issue_type", "description"],
        }
        
        # 不应传递给 LLM 的字段
        sensitive_fields = {
            "password", "credit_card", "id_number",
            "bank_account", "ssn", "api_key"
        }
        
        allowed = task_fields.get(task, list(data.keys()))
        
        return {
            k: v for k, v in data.items()
            if k in allowed and k not in sensitive_fields
        }
```

---

## 小结

| 概念 | 说明 |
|------|------|
| PII 检测 | 用正则等方式识别敏感信息 |
| 数据脱敏 | 替换敏感信息为掩码形式 |
| 对话保护 | 存储前自动脱敏处理 |
| 数据最小化 | 只传递完成任务所需的最少数据 |

> **下一节预告**：最后，我们来讨论如何确保 Agent 的行为符合人类的期望和价值观。

---

[下一节：17.5 Agent 行为的可控性与对齐 →](./05_alignment.md)
