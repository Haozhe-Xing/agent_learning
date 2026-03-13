# 数据连接与查询

> **本节目标**：实现安全的数据库连接和自然语言到 SQL 的转换。

---

## 安全的数据库连接

```python
import sqlite3
from contextlib import contextmanager

class SafeDatabaseConnector:
    """安全的数据库连接器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典格式
        try:
            yield conn
        finally:
            conn.close()
    
    def get_table_schemas(self) -> dict:
        """获取所有表的结构信息"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 获取所有表名
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
            
            schemas = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = [
                    {
                        "name": row[1],
                        "type": row[2],
                        "nullable": not row[3],
                        "primary_key": bool(row[5])
                    }
                    for row in cursor.fetchall()
                ]
                
                # 获取示例数据
                cursor.execute(f"SELECT * FROM {table} LIMIT 3")
                sample = [dict(row) for row in cursor.fetchall()]
                
                schemas[table] = {
                    "columns": columns,
                    "sample_data": sample
                }
            
            return schemas
    
    def execute_readonly(self, sql: str) -> list[dict]:
        """只执行只读查询（安全保障）"""
        # 安全检查：只允许 SELECT
        normalized = sql.strip().upper()
        if not normalized.startswith("SELECT"):
            raise PermissionError("只允许执行 SELECT 查询")
        
        # 禁止危险关键词
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", 
                     "ALTER", "CREATE", "TRUNCATE"]
        for keyword in dangerous:
            if keyword in normalized:
                raise PermissionError(f"查询中包含禁止的操作: {keyword}")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            return [dict(row) for row in cursor.fetchall()]
```

---

## 自然语言转 SQL（Text-to-SQL）

```python
class TextToSQL:
    """自然语言转 SQL"""
    
    def __init__(self, llm, db: SafeDatabaseConnector):
        self.llm = llm
        self.db = db
        self.schemas = db.get_table_schemas()
    
    async def convert(self, question: str) -> str:
        """将自然语言问题转为 SQL"""
        
        schema_desc = self._format_schemas()
        
        prompt = f"""你是一个 SQL 专家。根据用户的自然语言问题，生成对应的 SQL 查询。

数据库表结构：
{schema_desc}

用户问题：{question}

要求：
1. 只生成 SELECT 查询
2. 使用标准 SQL 语法
3. 只返回 SQL 语句，不要其他文字
4. 如果问题模糊，做合理假设
"""
        
        response = await self.llm.ainvoke(prompt)
        sql = response.content.strip()
        
        # 清理（移除可能的 markdown 代码块标记）
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1]
        if sql.endswith("```"):
            sql = sql.rsplit("```", 1)[0]
        
        return sql.strip()
    
    def _format_schemas(self) -> str:
        """格式化表结构描述"""
        lines = []
        for table, info in self.schemas.items():
            cols = ", ".join(
                f"{c['name']} ({c['type']})" for c in info["columns"]
            )
            lines.append(f"表 {table}: {cols}")
            
            if info["sample_data"]:
                sample = str(info["sample_data"][0])
                lines.append(f"  示例数据: {sample[:200]}")
        
        return "\n".join(lines)
```

---

## 使用示例

```python
async def demo():
    db = SafeDatabaseConnector("sales.db")
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    t2s = TextToSQL(llm, db)
    
    questions = [
        "上个月销售额最高的前 5 个产品是什么？",
        "按区域统计今年的销售总额",
        "哪些客户最近 3 个月没有下单？"
    ]
    
    for q in questions:
        sql = await t2s.convert(q)
        print(f"问题：{q}")
        print(f"SQL：{sql}")
        
        try:
            results = db.execute_readonly(sql)
            print(f"结果：{results[:3]}...")
        except Exception as e:
            print(f"执行出错：{e}")
        print()
```

---

## 小结

| 组件 | 功能 |
|------|------|
| SafeDatabaseConnector | 安全的只读数据库访问 |
| TextToSQL | 自然语言自动转换为 SQL |
| 安全检查 | 禁止 DELETE/UPDATE 等危险操作 |

---

[下一节：17.3 自动化分析与可视化 →](./03_analysis_visualization.md)
