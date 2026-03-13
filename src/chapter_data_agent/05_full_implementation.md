# 完整项目实现

> **本节目标**：整合所有组件，构建一个完整的智能数据分析 Agent。

---

## 完整实现

```python
"""
智能数据分析 Agent —— 完整实现
用自然语言完成数据分析的全流程
"""
import asyncio
from langchain_openai import ChatOpenAI

# 导入前面实现的组件
# 各组件的完整实现请参考对应章节：
# from db_connector import SafeDatabaseConnector   # → 17.2 节
# from text_to_sql import TextToSQL                # → 17.2 节
# from data_analyzer import DataAnalyzer           # → 17.3 节
# from chart_generator import ChartGenerator       # → 17.3 节
# from insight_generator import InsightGenerator   # → 17.3 节
# from report_generator import ReportGenerator     # → 17.4 节
# 提示：运行本节代码前，需先将 17.2-17.4 节的代码保存为独立模块


class SmartDataAnalyst:
    """智能数据分析 Agent"""
    
    def __init__(self, db_path: str):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.db = SafeDatabaseConnector(db_path)
        self.text2sql = TextToSQL(self.llm, self.db)
        self.analyzer = DataAnalyzer()
        self.chart_gen = ChartGenerator()
        self.insight_gen = InsightGenerator(self.llm)
        self.report_gen = ReportGenerator(self.llm)
    
    async def ask(self, question: str) -> str:
        """用自然语言提问，获取完整分析"""
        
        print(f"🤔 理解问题: {question}")
        
        # 1. 自然语言 → SQL
        print("📝 生成查询...")
        sql = await self.text2sql.convert(question)
        print(f"   SQL: {sql}")
        
        # 2. 执行查询
        print("🔍 查询数据...")
        try:
            data = self.db.execute_readonly(sql)
        except Exception as e:
            return f"❌ 查询出错: {e}"
        
        if not data:
            return "📭 查询没有返回结果，请换个问法试试。"
        
        print(f"   获得 {len(data)} 条数据")
        
        # 3. 统计分析
        print("📊 分析数据...")
        stats = self.analyzer.describe(data)
        
        # 4. 生成图表
        print("🎨 生成图表...")
        chart_path = self.chart_gen.auto_chart(data, question)
        
        # 5. 生成洞察
        print("💡 提取洞察...")
        insights = await self.insight_gen.generate_insights(
            data, stats, question
        )
        
        # 6. 生成报告
        print("📄 生成报告...")
        report = await self.report_gen.generate_report(
            question=question,
            sql_query=sql,
            data=data,
            stats=stats,
            insights=insights,
            chart_path=chart_path
        )
        
        # 保存报告
        filepath = self.report_gen.save_report(report)
        print(f"✅ 报告已保存: {filepath}")
        
        return report


async def main():
    """交互式数据分析"""
    import sys
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "example.db"
    
    print("📊 智能数据分析助手")
    print("=" * 40)
    print("用自然语言描述你的分析需求")
    print("输入 'quit' 退出\n")
    
    analyst = SmartDataAnalyst(db_path)
    
    # 展示可用的表
    schemas = analyst.db.get_table_schemas()
    print(f"📁 数据库中有 {len(schemas)} 张表:")
    for table, info in schemas.items():
        cols = [c['name'] for c in info['columns']]
        print(f"   • {table}: {', '.join(cols)}")
    print()
    
    while True:
        question = input("你的问题: ").strip()
        
        if question.lower() in ('quit', 'exit', 'q'):
            print("👋 再见！")
            break
        
        if not question:
            continue
        
        result = await analyst.ask(question)
        print(f"\n{result}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 使用效果

```
📊 智能数据分析助手
========================================
📁 数据库中有 3 张表:
   • orders: id, customer_id, product, amount, date, region
   • customers: id, name, email, city, register_date
   • products: id, name, category, price

你的问题: 哪个区域的订单金额最高？按区域排序
🤔 理解问题: 哪个区域的订单金额最高？按区域排序
📝 生成查询...
   SQL: SELECT region, SUM(amount) as total FROM orders GROUP BY region ORDER BY total DESC
🔍 查询数据...
   获得 4 条数据
📊 分析数据...
🎨 生成图表...
💡 提取洞察...
📄 生成报告...
✅ 报告已保存: report_20260312_140000.md
```

---

## 小结

| 步骤 | 组件 | 说明 |
|------|------|------|
| 理解 | TextToSQL | 自然语言 → SQL |
| 查询 | SafeDB | 安全执行只读查询 |
| 分析 | DataAnalyzer | 统计分析 |
| 可视化 | ChartGenerator | 自动图表 |
| 洞察 | InsightGenerator | LLM 生成洞察 |
| 报告 | ReportGenerator | 完整分析报告 |

> 🎓 **本章总结**：我们构建了一个"用自然语言做数据分析"的完整 Agent。从 Text-to-SQL 到自动可视化，展示了 Agent 在数据分析领域的强大应用。

---

[下一章：第18章 项目实战：多模态 Agent →](../chapter_multimodal/README.md)
