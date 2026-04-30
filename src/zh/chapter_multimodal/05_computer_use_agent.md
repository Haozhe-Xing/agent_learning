# 22.5 Computer Use 与 GUI Agent

> **本节目标**：掌握 Computer Use Agent 的核心架构与实现，理解 GUI 自动化在 2025-2026 年的最新进展。

---

## 从"对话"到"操作"：Agent 的终极形态

前面的多模态 Agent 本质上还是"对话式"的——用户问、Agent 答。但在真实工作中，我们经常需要 Agent **直接操作软件**：打开浏览器搜索信息、在 Excel 中填入数据、在 IDE 中修改代码、在操作系统中安装软件。

**Computer Use Agent（计算机使用智能体）** 正是为了解决这个问题而诞生的：Agent 不再只是"嘴巴说"，而是"动手做"——通过理解屏幕截图、计算点击坐标、执行键盘输入，像人一样操作计算机图形界面（GUI）。

> 📄 **里程碑事件**：
> - **2024 年 10 月**：Anthropic 发布 Claude 3.5 Sonnet 的 Computer Use beta，首次让主流大模型直接操作桌面计算机
> - **2025 年 1 月**：OpenAI 发布 Operator，基于 CUA（Computer Using Agent）模型实现浏览器自动化
> - **2025 年 3 月**：Google 发布 Mariner，让 Gemini 2.0 可以操作 Chrome 浏览器
> - **2025-2026 年**：开源社区涌现 SWE-Agent、OpenHands、OSAtlas 等框架

---

## Computer Use 的核心循环

Computer Use Agent 的工作方式与人操作计算机的流程几乎一致：

```
┌─────────────────────────────────────────────────┐
│                  Computer Use Loop               │
│                                                  │
│  用户指令："帮我打开浏览器搜索今天的天气"          │
│       ↓                                          │
│  ① 截屏（Screenshot）                            │
│       ↓                                          │
│  ② 视觉模型理解屏幕内容                           │
│       ↓                                          │
│  ③ 规划下一步操作（点击/输入/滚动）                │
│       ↓                                          │
│  ④ 执行操作（移动鼠标、点击、输入文字）            │
│       ↓                                          │
│  ⑤ 回到 ①，直到任务完成                          │
└─────────────────────────────────────────────────┘
```

这本质上是一个 **感知-思考-行动（Perceive-Think-Act）** 循环——只是感知的输入是屏幕截图，行动的输出是鼠标/键盘事件。

---

## Anthropic Computer Use 实战

### 基础调用

Anthropic 的 Computer Use 通过 Beta API 提供，核心是 `computer_20241022` 工具：

```python
import anthropic

client = anthropic.Anthropic()

# 定义 Computer Use 工具
computer_tool = {
    "type": "computer_20241022",
    "name": "computer",
    "display_width_px": 1920,
    "display_height_px": 1080,
    "display_number": 1,
}

# 其他辅助工具
bash_tool = {
    "type": "bash_20241022",
    "name": "bash",
}

text_editor_tool = {
    "type": "text_editor_20241022",
    "name": "str_replace_based_editor",
}


def run_computer_use(task: str, max_steps: int = 20) -> list[dict]:
    """运行 Computer Use Agent 执行任务
    
    Args:
        task: 用户的自然语言指令
        max_steps: 最大执行步数（防止无限循环）
    """
    messages = [{"role": "user", "content": task}]
    steps = []
    
    for step in range(max_steps):
        response = client.beta.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=[computer_tool, bash_tool, text_editor_tool],
            messages=messages,
            betas=["computer-use-2025-01-24"],
        )
        
        # 收集所有工具调用
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                print(f"🤖 Agent: {block.text}")
            elif block.type == "tool_use":
                tool_calls.append(block)
                steps.append({
                    "step": step + 1,
                    "tool": block.name,
                    "input": block.input
                })
        
        if not tool_calls:
            # Agent 认为任务已完成
            break
        
        # 执行工具调用并返回结果
        tool_results = []
        for tool_call in tool_calls:
            result = execute_tool_action(tool_call)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result,
            })
        
        # 更新对话历史
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
    
    return steps


def execute_tool_action(tool_call) -> list[dict]:
    """执行 Computer Use 工具调用
    
    在实际部署中，这里需要集成真实的屏幕控制：
    - pyautogui / pynput 控制鼠标键盘
    - Pillow 截取屏幕
    - 在沙箱环境中执行 bash 命令
    
    以下为模拟实现的核心逻辑。
    """
    action = tool_call.input
    
    if tool_call.name == "computer":
        action_type = action.get("action")
        
        if action_type == "screenshot":
            # 截屏并返回 base64 编码的图像
            import pyautogui
            from PIL import Image
            import io, base64
            
            screenshot = pyautogui.screenshot()
            buffer = io.BytesIO()
            screenshot.save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            return [{
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_b64,
                }
            }]
        
        elif action_type == "mouse_move":
            x, y = action["coordinate"]
            import pyautogui
            pyautogui.moveTo(x, y)
            return [{"type": "text", "text": f"鼠标移动到 ({x}, {y})"}]
        
        elif action_type == "left_click":
            x, y = action["coordinate"]
            import pyautogui
            pyautogui.click(x, y)
            return [{"type": "text", "text": f"左键点击 ({x}, {y})"}]
        
        elif action_type == "type":
            text = action["text"]
            import pyautogui
            pyautogui.typewrite(text, interval=0.05)
            return [{"type": "text", "text": f"输入文字: {text}"}]
        
        elif action_type == "key":
            keys = action["key"]
            import pyautogui
            pyautogui.hotkey(*keys.split("+"))
            return [{"type": "text", "text": f"按键: {keys}"}]
        
        elif action_type == "scroll":
            x, y = action.get("coordinate", (0, 0))
            delta = action.get("delta", 1)
            import pyautogui
            pyautogui.scroll(delta, x, y)
            return [{"type": "text", "text": f"滚动: delta={delta}"}]
    
    elif tool_call.name == "bash":
        # 在沙箱中执行 bash 命令
        import subprocess
        cmd = action.get("command", "")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, 
                text=True, timeout=30
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = "Command timed out after 30 seconds"
        return [{"type": "text", "text": output}]
    
    return [{"type": "text", "text": "Unknown action"}]


# 使用示例
steps = run_computer_use(
    "打开浏览器，搜索'2026年4月最新AI论文'，并把前3个结果保存到文件中"
)
print(f"\n总共执行了 {len(steps)} 步操作")
```

### 安全边界设计

Computer Use Agent 的安全风险远高于普通文本 Agent——它可以直接操作你的计算机。生产环境部署时**必须**设置严格的安全边界：

```python
class SafeComputerUseAgent:
    """带安全边界的 Computer Use Agent"""
    
    # 危险操作黑名单
    FORBIDDEN_COMMANDS = [
        "rm -rf", "del /s", "format", "mkfs",
        "shutdown", "reboot", "passwd",
        "curl *| *sh", "wget *| *sh",  # 管道到 shell
    ]
    
    # 敏感目录保护
    PROTECTED_PATHS = [
        "/etc/passwd", "/etc/shadow",
        "/.ssh/", "/.gnupg/",
        "C:\\Windows\\System32",
    ]
    
    # 允许操作的应用白名单（可选）
    ALLOWED_APPS = [
        "chrome", "firefox", "safari",       # 浏览器
        "code", "cursor", "vim",             # 编辑器
        "excel", "numbers",                  # 表格
        "terminal", "iterm", "cmd",          # 终端
    ]
    
    def __init__(self):
        self.screenshot_count = 0
        self.total_actions = 0
        self.max_actions_per_task = 50  # 单任务最大操作数
    
    def validate_bash_command(self, command: str) -> tuple[bool, str]:
        """验证 bash 命令是否安全"""
        cmd_lower = command.lower().strip()
        
        # 检查黑名单
        for forbidden in self.FORBIDDEN_COMMANDS:
            if forbidden.replace("*", "") in cmd_lower:
                return False, f"危险命令被拦截: {forbidden}"
        
        # 检查敏感路径
        for path in self.PROTECTED_PATHS:
            if path.lower() in cmd_lower:
                return False, f"敏感路径被保护: {path}"
        
        return True, "OK"
    
    def validate_click_target(self, x: int, y: int, 
                               screenshot_description: str) -> tuple[bool, str]:
        """验证点击目标是否安全
        
        通过屏幕截图描述判断点击区域是否合理
        """
        # 检查操作次数
        self.total_actions += 1
        if self.total_actions > self.max_actions_per_task:
            return False, "已超过单任务最大操作数，强制停止"
        
        # 检查是否点击了危险区域（如确认删除对话框）
        dangerous_keywords = ["删除所有", "格式化", "清空", "erase all"]
        if any(kw in screenshot_description for kw in dangerous_keywords):
            return False, "检测到危险操作界面，需要人工确认"
        
        return True, "OK"
    
    def get_confirmation(self, action: str, detail: str) -> bool:
        """对高风险操作请求人工确认"""
        print(f"\n⚠️  高风险操作请求: {action}")
        print(f"   详情: {detail}")
        confirm = input("   是否允许？(y/N): ").strip().lower()
        return confirm == "y"
```

---

## OpenAI Operator / CUA 模型

OpenAI 在 2025 年 1 月推出了 **CUA（Computer Using Agent）** 模型，与 Anthropic 的 Computer Use 形成竞争。CUA 的核心特点是**专门针对浏览器操作优化**：

```python
from openai import OpenAI

client = OpenAI()

def run_cua_agent(task: str) -> str:
    """使用 OpenAI CUA 模型操作浏览器
    
    CUA 模型的核心 API 是 response.action，
    它返回结构化的操作指令而非自由文本。
    """
    response = client.responses.create(
        model="computer-use-preview",
        tools=[{
            "type": "computer_use_preview",
            "display_width": 1280,
            "display_height": 720,
            "environment": "browser",  # CUA 专注浏览器场景
        }],
        input=[{
            "role": "user",
            "content": task
        }]
    )
    
    return response


# CUA 适用的浏览器自动化场景
CUA_BROWSER_TASKS = {
    "网页填写": "在 Google Forms 中填写问卷，姓名填'张三'，邮箱填'zhang@example.com'",
    "信息提取": "打开京东搜索'机械键盘'，提取前5个商品的价格和名称",
    "网页操作": "在 GitHub 上创建一个名为'my-agent'的新仓库",
    "表单提交": "在携程上搜索北京到上海的机票，日期选下周六",
}
```

---

## Browser Use / Web Agent：最先落地的 Computer Use 场景

在所有 Computer Use 场景中，**浏览器自动化（Browser Use / Web Agent）** 是最先商业化、也最容易规模化的方向。原因很简单：大量工作流都发生在浏览器里——搜索资料、填写表单、操作 SaaS 后台、订票、购物、CRM 录入、网页数据提取。

Web Agent 与通用 GUI Agent 的关系可以这样理解：

| 维度 | Web Agent / Browser Use | 通用 Computer Use Agent |
|------|-------------------------|--------------------------|
| **操作环境** | 浏览器页面 | 整个操作系统和任意应用 |
| **感知输入** | DOM、Accessibility Tree、截图、网络请求 | 截图、可访问性树、系统状态 |
| **动作空间** | 点击、输入、滚动、选择、导航、下载 | 鼠标、键盘、命令行、文件系统、应用切换 |
| **可验证性** | URL、DOM 状态、页面文本、表单值较易验证 | 依赖截图和系统状态，验证更难 |
| **安全风险** | 钓鱼页面、间接提示注入、越权提交 | 文件破坏、系统命令、跨应用误操作 |
| **代表产品** | OpenAI Operator、Google Mariner、Browser Use | Claude Computer Use、OpenHands、OSWorld Agent |

### Web Agent 的核心循环

> 用户目标 → 打开/搜索页面 → 读取 DOM + 截图 → 定位可交互元素 → 执行点击/输入/滚动 → 验证页面状态 → 继续下一步或提交结果

与传统爬虫不同，Web Agent 不是只"读网页"，而是能**操作网页**。与 RPA 不同，Web Agent 不依赖固定脚本，而是能根据页面变化动态决策。

### 为什么 Browser Use 是前沿重点？

1. **网页是现实世界 API 的外壳**：很多业务系统没有开放 API，但有网页界面。
2. **可执行环境相对标准化**：浏览器比桌面 OS 更容易沙箱化、录制轨迹和回放测试。
3. **评估基准更成熟**：WebArena、VisualWebArena、Mind2Web 等基准提供了可比较任务。
4. **安全挑战典型**：网页内容是外部不可信输入，容易触发间接提示注入。

### 工程实现建议

生产级 Web Agent 不应只依赖截图坐标，推荐采用混合感知：

```text
DOM / Accessibility Tree：定位按钮、输入框、链接
截图：理解视觉布局、广告遮挡、验证码、复杂组件
网络状态：判断页面是否加载完成、请求是否失败
浏览器上下文：记录 URL、Cookie、下载文件和标签页状态
```

这类 Agent 一定要配合后文的安全章节：网页内容不能直接当作高优先级指令，提交订单、发送消息、修改配置等高风险动作必须经过权限检查和人工确认。

---

## GUI Agent 的三种架构

从实现角度，GUI Agent 可以分为三种架构：

### 架构一：截图 + 坐标（Screenshot + Coordinate）

Anthropic Computer Use 和 OpenAI CUA 采用此架构。Agent 看截图、输出坐标。

```python
# 核心循环伪代码
class ScreenshotCoordinateAgent:
    """截图+坐标架构：最通用的 GUI Agent"""
    
    def run(self, task: str):
        while not self.is_done(task):
            # 1. 截屏
            screenshot = self.take_screenshot()
            
            # 2. LLM 分析截图并输出操作
            action = self.llm.decide(
                task=task,
                screenshot=screenshot,
                history=self.action_history
            )
            
            # 3. 执行操作
            if action.type == "click":
                self.mouse.click(action.x, action.y)
            elif action.type == "type":
                self.keyboard.type(action.text)
            elif action.type == "scroll":
                self.mouse.scroll(action.delta)
            
            self.action_history.append(action)
```

**优点**：通用性强，任何 GUI 都能操作  
**缺点**：坐标精度依赖屏幕分辨率，对小型元素容易误点

### 架构二：Accessibility Tree（无障碍树）

利用操作系统的 Accessibility API 获取界面元素的树状结构，精确匹配元素而非坐标。

```python
import subprocess
import json

class AccessibilityTreeAgent:
    """无障碍树架构：精确但需要平台支持"""
    
    def get_accessibility_tree(self) -> dict:
        """获取当前窗口的无障碍树
        
        macOS: 使用 Accessibility API
        Windows: 使用 UI Automation
        Linux: 使用 ATK/AT-SPI
        """
        # macOS 示例：通过 Swift/Python 获取
        script = '''
        import ApplicationServices
        let app = AXUIElementCreateApplication(pid)
        var value: AnyObject?
        AXUIElementCopyAttributeValue(app, kAXChildrenAttribute as CFString, &value)
        '''
        # 实际实现需要平台特定的 Accessibility 框架
        pass
    
    def click_element(self, element_description: str):
        """通过元素描述精确定位并点击
        
        例如：点击"提交按钮"而非"坐标(520, 340)"
        """
        tree = self.get_accessibility_tree()
        element = self._find_element(tree, element_description)
        
        if element:
            # 直接通过 Accessibility API 执行操作
            # 不需要计算坐标，不依赖屏幕分辨率
            self._perform_action(element, "click")
        else:
            # 回退到截图+坐标方式
            self._fallback_screenshot_click(element_description)
```

**优点**：精确匹配元素，不受分辨率影响  
**缺点**：需要平台支持，部分应用不暴露 Accessibility 信息

### 架构三：混合架构（Hybrid）

结合截图理解和 Accessibility Tree，先精确定位元素，截屏辅助理解上下文：

```python
class HybridGUIAgent:
    """混合架构：先精确匹配，再截图验证"""
    
    def execute_action(self, task: str, action_plan: dict):
        """执行一个操作步骤"""
        
        # 1. 先尝试 Accessibility Tree 精确定位
        element = self.acc_tree.find(action_plan["target"])
        
        if element and element.is_visible():
            # 精确定位成功，执行操作
            result = element.perform(action_plan["action"])
        else:
            # 精确定位失败，回退到截图+坐标
            screenshot = self.take_screenshot()
            coords = self.llm.locate_element(
                screenshot=screenshot,
                element_description=action_plan["target"]
            )
            self.mouse.click(coords.x, coords.y)
            result = "clicked_via_screenshot"
        
        # 2. 截图验证操作结果
        verification = self.take_screenshot()
        success = self.llm.verify_action(
            before=self.last_screenshot,
            after=verification,
            expected=action_plan["expected_result"]
        )
        
        return {"action": action_plan, "success": success}
```

### 三种架构对比

| 架构 | 精确度 | 通用性 | 平台依赖 | 代表 |
|------|--------|--------|---------|------|
| 截图+坐标 | 中 | 高 | 低 | Claude Computer Use, OpenAI CUA |
| Accessibility Tree | 高 | 中 | 高 | UFO (Microsoft), OSAtlas |
| 混合架构 | 高 | 高 | 中 | OpenHands, SWE-Agent |

---

## 开源 GUI Agent 框架

### OpenHands（原 OpenDevin）

OpenHands 是 2025 年最受欢迎的开源编程 Agent 框架之一，支持多种交互模式：

```python
# OpenHands 架构概览（伪代码，展示核心设计）
# 实际使用请参考：https://github.com/All-Hands-AI/OpenHands

"""
OpenHands 的核心设计：
1. 沙箱环境：每个任务在独立 Docker 容器中执行
2. 多模态交互：支持 bash、文件编辑、浏览器操作
3. Agent 循环：观察 → 推理 → 行动 → 验证

Action 类型：
- CmdRunAction: 执行 bash 命令
- FileWriteAction: 写入文件
- FileEditAction: 编辑文件（SED 格式）
- BrowseInteractiveAction: 浏览器交互
- MessageAction: 与用户对话
"""

# OpenHands 的文件编辑策略（Search-Replace Pattern）
SEARCH_REPLACE_EXAMPLE = """
<<< SEARCH
 def calculate_sum(numbers):
     total = 0
     for n in numbers:
         total += n
     return total
=======
 def calculate_sum(numbers):
     """计算列表中所有数字的总和"""
     return sum(numbers)
>>> REPLACE
"""
# 这种 Search-Replace 编辑模式比行号编辑更鲁棒
# Claude Code 也采用了类似的编辑策略
```

### SWE-Agent

Princeton 开发的 SWE-Agent 专注于解决 GitHub Issue，在 SWE-bench 上表现优异：

```python
# SWE-Agent 的核心创新：Agent-Computer Interface (ACI)
# 它为 LLM 定制了一套专用的命令行工具

SWE_AGENT_COMMANDS = {
    "find_file": "在仓库中搜索文件名",
    "open": "打开文件并显示行号",
    "search_dir": "在目录中搜索字符串",
    "search_file": "在文件中搜索字符串",
    "edit": "替换文件中的内容（行号范围）",
    "insert": "在指定行号后插入内容",
    "goto": "跳转到文件的指定行",
    "submit": "提交修改",
}

# SWE-Agent 的 ACI 设计哲学：
# 不是让 LLM 适应计算机的界面，而是为 LLM 定制界面
# 这与 Computer Use 的"让 LLM 学会操作人类界面"形成互补
```

---

## 评估基准

GUI Agent 的评估是目前活跃的研究方向：

| 基准 | 评估场景 | 核心指标 | 说明 |
|------|---------|---------|------|
| **OSWorld** | 桌面操作系统 | 任务完成率 | 369 个真实桌面任务，覆盖 Ubuntu/Windows/macOS |
| **VisualWebArena** | Web 应用 | 端到端准确率 | 电商、论坛、CMS 三类网站操作 |
| **WebArena** | Web 应用 | 端到端准确率 | VisualWebArena 的纯文本版本 |
| **SWE-bench** | 代码仓库 | Issue 解决率 | 解决真实 GitHub Issue |
| **Mind2Web** | Web 操作 | 元素定位准确率 | 2000+ 真实网页操作任务 |
| **AndroidWorld** | Android 手机 | 任务完成率 | 116 个 Android 操作任务 |

> 💡 **前沿进展**：截至 2026 年 4 月，OSWorld 上的最佳 Agent 得分约 12.5%（人类约 72%），VisualWebArena 上约 38%（人类约 88%）。GUI Agent 仍处于早期阶段，但进步速度惊人——2024 年中 OSWorld 最佳得分仅约 5%。

---

## Computer Use Agent 的最佳实践

### 1. 任务分解

复杂的 GUI 操作应分解为小步骤：

```python
# 好的做法：将复杂任务分解
GOOD_PROMPT = """
请在 Excel 中完成以下操作（每步完成后截屏确认）：
1. 打开文件 'sales_2026.xlsx'
2. 选中 A1:D20 区域
3. 点击"插入"→"图表"→"柱状图"
4. 将图表标题改为"2026年销售数据"
5. 保存文件
"""

# 不好的做法：一次性要求太多
BAD_PROMPT = "帮我用 Excel 处理一下那个销售表格"
```

### 2. 错误恢复

GUI 操作容易出错（弹窗、加载延迟、元素位置变化），需要健壮的错误恢复：

```python
class RobustGUIAgent:
    """带错误恢复的 GUI Agent"""
    
    MAX_RETRIES = 3
    WAIT_AFTER_ACTION = 1.0  # 操作后等待时间（秒）
    
    async def execute_with_retry(self, action: dict) -> bool:
        """带重试的操作执行"""
        for attempt in range(self.MAX_RETRIES):
            try:
                # 执行操作
                self._perform_action(action)
                
                # 等待 UI 更新
                await asyncio.sleep(self.WAIT_AFTER_ACTION)
                
                # 验证操作是否生效
                screenshot = self.take_screenshot()
                success = await self.llm_verify(
                    screenshot=screenshot,
                    expected_state=action.get("expected_result")
                )
                
                if success:
                    return True
                
                print(f"⚠️ 操作未生效，重试 {attempt + 1}/{self.MAX_RETRIES}")
                
            except Exception as e:
                print(f"❌ 操作出错: {e}")
                # 截屏诊断
                self.take_screenshot("error_debug.png")
        
        return False
```

### 3. 沙箱隔离

**永远不要在生产机器上直接运行 Computer Use Agent。** 必须使用沙箱环境：

```python
# 推荐的沙箱方案
SANDBOX_OPTIONS = {
    "Docker": {
        "优点": "成熟、隔离性好、易于复现",
        "适用": "Linux 服务器、CI/CD 环境",
        "工具": "Docker Compose + VNC（可视化）",
    },
    "E2B": {
        "优点": "专为 AI Agent 设计、秒级启动、安全",
        "适用": "云端代码执行、编程 Agent",
        "工具": "E2B Code Interpreter SDK",
    },
    "VM (QEMU/VirtualBox)": {
        "优点": "完整 OS 隔离、支持 GUI 渲染",
        "适用": "桌面应用自动化测试",
        "工具": "Vagrant + VirtualBox",
    },
    "Modal/RunPod": {
        "优点": "GPU 支持、Serverless、按需计费",
        "适用": "需要 GPU 的视觉推理场景",
        "工具": "Modal SDK / RunPod Serverless",
    },
}
```

---

## 小结

| 概念 | 说明 |
|------|------|
| Computer Use Agent | 通过截屏理解和鼠标/键盘操作控制计算机的 Agent |
| Browser Use / Web Agent | 专注浏览器环境的 Agent，能读取并操作网页 |
| 核心循环 | 截屏/DOM → 理解 → 规划操作 → 执行 → 验证 → 重复 |
| 主流产品 | Anthropic Computer Use、OpenAI Operator/CUA、Google Mariner |
| 三种架构 | 截图+坐标（通用）、Accessibility Tree（精确）、混合架构 |
| 开源框架 | OpenHands、SWE-Agent、OSAtlas、Browser Use |
| 评估基准 | OSWorld、VisualWebArena、WebArena、Mind2Web、SWE-bench |
| 安全红线 | 必须沙箱隔离、操作白名单、人工确认高风险操作 |

> 📄 **延伸阅读**：
> - Anthropic. "Developing Computer Use." Claude Documentation, 2024.
> - OpenAI. "Operator & CUA." OpenAI Blog, 2025.
> - Xue et al. "OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments." ICLR, 2025.
> - Wang et al. "OpenHands: An Open Platform for AI Software Developers." arXiv:2407.16741, 2024.

---

[22.6 视频理解与多模态 RAG](./06_video_and_multimodal_rag.md)
