# 2.1 Python 环境与依赖管理

Agent 项目依赖众多第三方库，版本冲突是常见的"头疼问题"。使用虚拟环境可以为每个项目创建独立的依赖空间，彻底避免冲突。

## 为什么需要虚拟环境？

![虚拟环境方案对比](../svg/chapter_setup_01_venv_compare.svg)

## 方案一：uv（推荐，Python 包管理新标准）

`uv` 是 Rust 编写的超快 Python 包管理器，比 `pip` 快 10-100 倍：

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装
uv --version

# 创建新项目
mkdir my_agent_project && cd my_agent_project
uv init

# 创建虚拟环境（自动使用项目目录的 .venv）
uv venv

# 激活虚拟环境
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate      # Windows

# 安装依赖（比 pip 快得多！）
uv add openai langchain python-dotenv

# 查看已安装的包
uv pip list

# 生成锁定文件（确保团队环境一致）
uv lock

# 从锁定文件安装（CI/CD 或团队协作）
uv sync
```

**uv 的 pyproject.toml：**

```toml
# pyproject.toml（uv init 自动生成）
[project]
name = "my-agent-project"
version = "0.1.0"
description = "My first Agent project"
requires-python = ">=3.11"
dependencies = [
    "openai>=1.60.0",
    "langchain>=0.3.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "ruff>=0.8.0",
]
```

## 方案二：conda（数据科学友好）

如果你已经在使用 Anaconda/Miniconda：

```bash
# 创建专用环境
conda create -n agent_dev python=3.11

# 激活环境
conda activate agent_dev

# 安装包
conda install -c conda-forge openai
pip install langchain python-dotenv  # conda 没有的包用 pip

# 导出环境配置
conda env export > environment.yml

# 从配置文件重建环境（团队协作）
conda env create -f environment.yml
```

## 方案三：venv（内置，无需安装）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活
source .venv/bin/activate    # Linux/Mac
# .venv\Scripts\activate.bat  # Windows cmd
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# 安装依赖
pip install openai langchain python-dotenv

# 保存依赖列表
pip freeze > requirements.txt

# 从文件安装
pip install -r requirements.txt
```

## 推荐的项目结构

```
my_agent_project/
├── .venv/                  # 虚拟环境（不提交到 git）
├── .env                    # API Keys（不提交到 git！）
├── .env.example            # Key 模板（提交到 git）
├── .gitignore              # 排除 .venv 和 .env
├── pyproject.toml          # 项目配置（uv）
├── requirements.txt        # 依赖列表（pip）
├── src/
│   ├── __init__.py
│   ├── agent.py            # Agent 主逻辑
│   ├── tools.py            # 工具定义
│   └── config.py           # 配置管理
├── tests/
│   └── test_agent.py
└── notebooks/
    └── experiments.ipynb   # 实验性代码
```

**标准 .gitignore：**

```gitignore
# 虚拟环境
.venv/
venv/
env/

# API Keys（重要！）
.env
.env.local
*.env

# Python 缓存
__pycache__/
*.pyc
*.pyo
.pytest_cache/

# 编辑器
.vscode/settings.json
.idea/

# 数据文件
*.db
*.sqlite
data/raw/
```

## Python 版本管理

对于需要管理多个 Python 版本的开发者：

```bash
# 使用 pyenv（推荐）
# 安装 pyenv
curl https://pyenv.run | bash

# 安装指定版本
pyenv install 3.12.8
pyenv install 3.13.2

# 设置全局/本地版本
pyenv global 3.12.8         # 全局
pyenv local 3.13.2          # 当前目录

# uv 也支持自动管理 Python 版本
uv python install 3.13
uv python pin 3.13  # 为当前项目固定版本
```

## 快速验证环境

```python
# check_env.py：运行这个脚本验证环境是否正确
import sys
import pkg_resources

print(f"Python 版本：{sys.version}")
print(f"Python 路径：{sys.executable}")
print()

required_packages = [
    "openai",
    "langchain",
    "python-dotenv",
]

print("依赖检查：")
for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"  ✅ {package} {version}")
    except pkg_resources.DistributionNotFound:
        print(f"  ❌ {package} 未安装")

# 验证 API Key
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    # 只显示前几位，不暴露完整 Key
    print(f"\n✅ OPENAI_API_KEY 已设置（{api_key[:8]}...）")
else:
    print("\n❌ OPENAI_API_KEY 未设置，请检查 .env 文件")
```

```bash
# 运行检查
python check_env.py
```

---

## 小结

| 工具 | 适合场景 | 速度 |
|------|---------|------|
| `uv` | 新项目，追求速度 | ⚡⚡⚡⚡⚡ |
| `conda` | 数据科学，需要非 Python 依赖 | ⚡⚡ |
| `venv` | 简单项目，不想装额外工具 | ⚡⚡⚡ |

推荐使用 `uv`——它快、简单、现代，正在成为 Python 包管理的新标准。

---

*下一节：[2.2 关键库安装：LangChain、OpenAI SDK 等](./02_install_libs.md)*
