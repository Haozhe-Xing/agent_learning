# 权限控制与沙箱隔离

> **本节目标**：学会为 Agent 设计最小权限体系和安全的执行环境。

---

## 最小权限原则

Agent 应该只拥有完成任务所需的最小权限——就像公司不应该给每个员工发一把万能钥匙。

![Agent 权限控制与沙箱隔离架构](../svg/chapter_security_03_permission.svg)

```python
from enum import Flag, auto
from dataclasses import dataclass

class Permission(Flag):
    """Agent 权限定义"""
    NONE = 0
    READ_FILE = auto()       # 读取文件
    WRITE_FILE = auto()      # 写入文件
    EXECUTE_CODE = auto()    # 执行代码
    NETWORK_ACCESS = auto()  # 网络访问
    DATABASE_READ = auto()   # 读取数据库
    DATABASE_WRITE = auto()  # 写入数据库
    SEND_EMAIL = auto()      # 发送邮件
    
    # 预定义的权限组合
    READONLY = READ_FILE | DATABASE_READ
    STANDARD = READONLY | WRITE_FILE | NETWORK_ACCESS
    FULL = STANDARD | EXECUTE_CODE | DATABASE_WRITE | SEND_EMAIL


@dataclass
class PermissionPolicy:
    """权限策略"""
    agent_name: str
    permissions: Permission
    allowed_paths: list[str] = None     # 允许访问的文件路径
    allowed_domains: list[str] = None   # 允许访问的网络域名
    max_file_size: int = 10 * 1024 * 1024  # 最大文件大小（10MB）
    
    def check(self, action: str, resource: str = None) -> bool:
        """检查是否有权限执行某个操作"""
        perm_map = {
            "read_file": Permission.READ_FILE,
            "write_file": Permission.WRITE_FILE,
            "execute": Permission.EXECUTE_CODE,
            "http_request": Permission.NETWORK_ACCESS,
            "db_read": Permission.DATABASE_READ,
            "db_write": Permission.DATABASE_WRITE,
            "send_email": Permission.SEND_EMAIL,
        }
        
        required = perm_map.get(action)
        if required is None:
            return False
        
        if not (self.permissions & required):
            return False
        
        # 检查资源级别的权限
        if action in ("read_file", "write_file") and resource:
            if self.allowed_paths:
                return any(
                    resource.startswith(p) for p in self.allowed_paths
                )
        
        if action == "http_request" and resource:
            if self.allowed_domains:
                from urllib.parse import urlparse
                domain = urlparse(resource).hostname
                return domain in self.allowed_domains
        
        return True


# 使用示例
customer_service_policy = PermissionPolicy(
    agent_name="customer_service",
    permissions=Permission.READONLY | Permission.NETWORK_ACCESS,
    allowed_paths=["/data/faq/", "/data/products/"],
    allowed_domains=["api.internal.com"]
)

# 检查权限
print(customer_service_policy.check("read_file", "/data/faq/guide.md"))  # True
print(customer_service_policy.check("write_file", "/etc/passwd"))  # False
print(customer_service_policy.check("execute"))  # False
```

---

## 安全工具包装器

在工具执行前后加上安全检查：

```python
import functools

def secure_tool(policy: PermissionPolicy):
    """安全工具装饰器 —— 为工具添加权限检查"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tool_name = func.__name__
            
            # 权限检查
            action = _infer_action(tool_name)
            resource = kwargs.get("path") or kwargs.get("url")
            
            if not policy.check(action, resource):
                return {
                    "error": f"权限不足: {tool_name} 需要 {action} 权限",
                    "allowed": False
                }
            
            # 执行工具
            try:
                result = func(*args, **kwargs)
                
                # 记录审计日志
                _log_tool_execution(
                    agent=policy.agent_name,
                    tool=tool_name,
                    args=kwargs,
                    success=True
                )
                
                return result
                
            except Exception as e:
                _log_tool_execution(
                    agent=policy.agent_name,
                    tool=tool_name,
                    args=kwargs,
                    success=False,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


def _infer_action(tool_name: str) -> str:
    """从工具名推断需要的权限"""
    action_keywords = {
        "read": "read_file",
        "write": "write_file",
        "save": "write_file",
        "execute": "execute",
        "run": "execute",
        "fetch": "http_request",
        "search": "http_request",
        "query": "db_read",
        "insert": "db_write",
        "delete": "db_write",
        "email": "send_email",
    }
    
    for keyword, action in action_keywords.items():
        if keyword in tool_name.lower():
            return action
    
    return "unknown"


def _log_tool_execution(**kwargs):
    """记录工具执行日志（简化版）"""
    import json, datetime
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        **kwargs
    }
    print(f"[AUDIT] {json.dumps(log_entry, ensure_ascii=False)}")
```

---

## 代码执行沙箱

如果 Agent 需要执行代码，必须在隔离环境中运行：

```python
import subprocess
import tempfile
import os

class CodeSandbox:
    """代码执行沙箱"""
    
    def __init__(
        self,
        timeout: int = 10,
        max_memory_mb: int = 256,
        allowed_imports: list[str] = None
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_imports = allowed_imports or [
            "math", "json", "datetime", "re",
            "collections", "itertools", "functools",
            "statistics", "random", "string"
        ]
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """在执行前验证代码安全性"""
        import ast
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"语法错误: {e}"
        
        # 检查危险操作
        dangerous_calls = {
            "eval", "exec", "compile",
            "__import__", "globals", "locals",
            "getattr", "setattr", "delattr",
        }
        
        dangerous_modules = {
            "os", "sys", "subprocess", "shutil",
            "socket", "http", "urllib",
        }
        
        for node in ast.walk(tree):
            # 检查函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_calls:
                        return False, f"禁止调用: {node.func.id}()"
            
            # 检查 import
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in dangerous_modules:
                        return False, f"禁止导入: {module_name}"
                    if (self.allowed_imports and 
                        module_name not in self.allowed_imports):
                        return False, f"不在白名单中: {module_name}"
            
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in dangerous_modules:
                        return False, f"禁止导入: {module_name}"
        
        return True, "代码检查通过"
    
    def execute(self, code: str) -> dict:
        """在沙箱中执行代码"""
        
        # 先验证
        is_safe, message = self.validate_code(code)
        if not is_safe:
            return {
                "success": False,
                "error": message,
                "output": ""
            }
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # 在子进程中执行（带资源限制）
            result = subprocess.run(
                ["python3", temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    "PATH": "/usr/bin:/usr/local/bin",
                    "HOME": tempfile.gettempdir(),
                }
            )
            
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr if result.returncode != 0 else "",
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"执行超时（{self.timeout}秒）",
                "output": ""
            }
        finally:
            os.unlink(temp_path)


# 使用示例
sandbox = CodeSandbox(timeout=5)

# 安全代码
result = sandbox.execute("""
import math
print(f"圆周率 = {math.pi:.10f}")
print(f"平方根2 = {math.sqrt(2):.10f}")
""")
print(result)  # {"success": True, "output": "圆周率 = ...\n", "error": ""}

# 危险代码会被拦截
result = sandbox.execute("""
import os
os.system("rm -rf /")
""")
print(result)  # {"success": False, "error": "禁止导入: os", "output": ""}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| 最小权限 | Agent 只拥有完成任务所需的最小权限 |
| 权限策略 | 定义允许的操作和可访问的资源 |
| 安全包装 | 在工具执行前后加权限检查和审计日志 |
| 代码沙箱 | 在隔离环境中执行不受信任的代码 |
| 代码验证 | 执行前通过 AST 分析检查危险操作 |

> **下一节预告**：Agent 可能会接触到用户的敏感数据，如何保护这些数据？

---

[下一节：17.4 敏感数据保护 →](./04_data_protection.md)
