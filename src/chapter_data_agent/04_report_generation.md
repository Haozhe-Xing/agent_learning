# 报告生成与导出

> **本节目标**：将分析结果自动整合为结构化的 Markdown 报告。

![报告自动生成流水线](../svg/chapter_data_04_report.svg)

---

## 报告生成器

```python
from datetime import datetime

class ReportGenerator:
    """分析报告生成器"""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def generate_report(
        self,
        question: str,
        sql_query: str,
        data: list[dict],
        stats: dict,
        insights: list[str],
        chart_path: str = None
    ) -> str:
        """生成完整的分析报告"""
        
        report = f"""# 📊 数据分析报告

> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}

## 分析问题

{question}

## 查询方式

```sql
{sql_query}
```

## 数据概览

- 数据量：{len(data)} 条记录
- 涉及字段：{', '.join(data[0].keys()) if data else '无'}
"""
        
        # 添加统计信息
        if stats.get("numeric_stats"):
            report += "\n## 统计摘要\n\n"
            report += "| 指标 | 均值 | 中位数 | 最小值 | 最大值 |\n"
            report += "|------|------|--------|--------|--------|\n"
            
            for col, s in stats["numeric_stats"].items():
                report += (
                    f"| {col} | {s['mean']:,.2f} | {s['median']:,.2f} "
                    f"| {s['min']:,.2f} | {s['max']:,.2f} |\n"
                )
        
        # 添加图表
        if chart_path:
            report += f"\n## 可视化\n\n![分析图表]({chart_path})\n"
        
        # 添加洞察
        if insights:
            report += "\n## 关键洞察\n\n"
            for i, insight in enumerate(insights, 1):
                report += f"{i}. {insight}\n"
        
        # 让 LLM 生成总结和建议
        summary = await self._generate_summary(question, insights, stats)
        report += f"\n## 总结与建议\n\n{summary}\n"
        
        return report
    
    async def _generate_summary(
        self, question, insights, stats
    ) -> str:
        """LLM 生成总结"""
        prompt = f"""基于以下分析结果，写一段简洁的总结和行动建议。

分析问题：{question}
关键洞察：{insights}
统计信息：{str(stats)[:500]}

要求：2-3 段话，包含核心发现和具体可执行的建议。"""
        
        response = await self.llm.ainvoke(prompt)
        return response.content
    
    def save_report(self, report: str, filename: str = None) -> str:
        """保存报告到文件"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"report_{timestamp}.md"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return filename
```

---

## 多格式导出

除了 Markdown，生产环境中通常需要支持多种导出格式：

```python
import json
from pathlib import Path

class MultiFormatExporter:
    """支持多种格式的报告导出器"""
    
    def export(self, report: str, data: list[dict], format: str, filename: str = None) -> str:
        """导出报告为指定格式"""
        exporters = {
            "markdown": self._export_markdown,
            "html": self._export_html,
            "json": self._export_json,
            "csv": self._export_csv,
        }
        
        exporter = exporters.get(format)
        if not exporter:
            raise ValueError(f"不支持的格式: {format}，可选: {list(exporters.keys())}")
        
        return exporter(report, data, filename)
    
    def _export_markdown(self, report: str, data: list[dict], filename: str = None) -> str:
        filename = filename or f"report_{self._timestamp()}.md"
        Path(filename).write_text(report, encoding="utf-8")
        return filename
    
    def _export_html(self, report: str, data: list[dict], filename: str = None) -> str:
        """将 Markdown 报告转为 HTML"""
        filename = filename or f"report_{self._timestamp()}.html"
        
        # 使用简单的 Markdown → HTML 转换
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>数据分析报告</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 0 auto; padding: 40px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background: #f5f5f5; font-weight: 600; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 4px; }}
        pre {{ background: #1e1e1e; color: #d4d4d4; padding: 16px; border-radius: 8px; overflow-x: auto; }}
        img {{ max-width: 100%; border-radius: 8px; }}
        blockquote {{ border-left: 4px solid #3498db; margin: 0; padding: 8px 16px; background: #f8f9fa; }}
    </style>
</head>
<body>
{self._md_to_html(report)}
</body>
</html>"""
        
        Path(filename).write_text(html_content, encoding="utf-8")
        return filename
    
    def _export_json(self, report: str, data: list[dict], filename: str = None) -> str:
        """导出结构化 JSON 数据"""
        filename = filename or f"report_{self._timestamp()}.json"
        
        output = {
            "report_text": report,
            "data": data,
            "generated_at": self._timestamp(),
        }
        
        Path(filename).write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return filename
    
    def _export_csv(self, report: str, data: list[dict], filename: str = None) -> str:
        """导出数据为 CSV"""
        import csv
        
        filename = filename or f"data_{self._timestamp()}.csv"
        
        if not data:
            Path(filename).write_text("", encoding="utf-8")
            return filename
        
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        
        return filename
    
    def _timestamp(self) -> str:
        return datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def _md_to_html(self, md: str) -> str:
        """简易 Markdown → HTML 转换"""
        try:
            import markdown
            return markdown.markdown(md, extensions=["tables", "fenced_code"])
        except ImportError:
            # 如果没有安装 markdown 库，返回 <pre> 包裹的原文
            return f"<pre>{md}</pre>"
```

---

## 定时报告与自动化

在生产环境中，数据分析报告通常需要定时生成并自动发送：

```python
import asyncio
from datetime import datetime

class ScheduledReporter:
    """定时报告生成器"""
    
    def __init__(self, agent, report_gen, exporter):
        self.agent = agent
        self.report_gen = report_gen
        self.exporter = exporter
        self.schedules = []
    
    def add_schedule(self, name: str, question: str, cron: str, recipients: list[str]):
        """添加定时报告任务
        
        示例: add_schedule(
            name="每日销售日报",
            question="统计今天各区域的销售额和订单量",
            cron="0 18 * * *",  # 每天18点
            recipients=["manager@company.com"]
        )
        """
        self.schedules.append({
            "name": name,
            "question": question,
            "cron": cron,
            "recipients": recipients,
        })
    
    async def run_report(self, schedule: dict) -> str:
        """执行一次报告生成"""
        print(f"📊 正在生成报告: {schedule['name']}")
        
        # 1. 用 Agent 执行分析
        result = await self.agent.analyze(schedule["question"])
        
        # 2. 生成报告
        report = await self.report_gen.generate_report(
            question=schedule["question"],
            sql_query=result.sql_query,
            data=result.data,
            stats=result.stats,
            insights=result.insights,
            chart_path=result.chart_path,
        )
        
        # 3. 导出为 HTML
        html_path = self.exporter.export(report, result.data, "html")
        
        print(f"✅ 报告已生成: {html_path}")
        return html_path
```

---

## 使用示例

```python
async def generate_sales_report():
    """生成销售分析报告的完整流程"""
    from langchain_openai import ChatOpenAI
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # 模拟数据
    data = [
        {"region": "华东", "sales": 1250000, "growth": 12.5},
        {"region": "华北", "sales": 980000, "growth": 8.3},
        {"region": "华南", "sales": 1100000, "growth": 15.2},
        {"region": "西部", "sales": 650000, "growth": 22.1},
    ]
    
    # 1. 分析
    analyzer = DataAnalyzer()
    stats = analyzer.describe(data)
    
    # 2. 可视化
    chart_gen = ChartGenerator()
    chart_path = chart_gen.auto_chart(data, "各区域销售额对比")
    
    # 3. 生成洞察
    insight_gen = InsightGenerator(llm)
    insights = await insight_gen.generate_insights(
        data, stats, "分析各区域销售表现"
    )
    
    # 4. 生成报告
    report_gen = ReportGenerator(llm)
    report = await report_gen.generate_report(
        question="分析各区域销售表现和增长趋势",
        sql_query="SELECT region, sales, growth FROM sales_summary",
        data=data,
        stats=stats,
        insights=insights,
        chart_path=chart_path
    )
    
    # 5. 保存
    filepath = report_gen.save_report(report)
    print(f"📄 报告已保存到: {filepath}")
```

---

## 小结

| 功能 | 说明 |
|------|------|
| 报告模板 | Markdown 格式，包含统计、图表、洞察 |
| LLM 总结 | 自动生成核心发现和行动建议 |
| 文件导出 | 保存为 .md 文件 |

---

[下一节：20.5 完整项目实现 →](./05_full_implementation.md)
