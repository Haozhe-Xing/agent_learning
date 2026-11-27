# 16A.4 实战：修复一个排名 Bug

本练习改编自 Damien Sileo 的 MIT 实操课。原课从「并列的正确答案可能被误判」切入，再让读者用终端 Agent、只读审阅员和独立检查找问题。本节把场景缩成两个 Python 文件，便于观察补丁与验收结果。[上游练习](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/GUIDED.md) · [本书示例](https://github.com/Haozhe-Xing/agent_learning/tree/main/examples/opencode-workshop)

## 准备一个可随时重来的工作区

从本书 Git 仓库根目录运行。下面以 macOS/Linux shell 为例，Windows 用户可在 WSL 使用同一组命令。复制时包含隐藏的 `.opencode/` 目录。

```bash
workshop_dir="$(mktemp -d)"
cp -R examples/opencode-workshop/. "$workshop_dir/"
cd "$workshop_dir"
git init
git add .
git -c user.name="OpenCode Workshop" -c user.email=workshop@example.com commit -m "Save workshop baseline"
python3 -B check_ranking.py
```

最后一条命令应在并列用例报错。分数最高的标签应获胜。如果两项并列，`alpha` 应在 `zeta` 前，无论它们在输入中的顺序如何。验收脚本还检查了非并列情形。

## 先计划，再仅修改实现

1. 运行 `opencode`，连接模型，用 `/agents` 选 **Plan**。附上两个文件并问：「为什么 `@check_ranking.py` 的并列用例失败？指出 `@ranking.py` 中的原因，提出最小改法。」
2. 在修改前可以先执行下文的只读审阅，记录这个缺陷的 `VERDICT`。
3. 用 `/agents` 切到 **Build**。要求它仅修改 `ranking.py`，保持签名不变。检查审批里显示的文件名，只批准预期改动。
4. 在自己的终端运行 `git diff -- ranking.py`、`git status --short` 和 `python3 -B check_ranking.py`。预期只出现 `ranking.py` 的修改，检查输出 `Ranking checks passed.`。如果想练习撤回，在提交补丁前试用 `/undo` 和 `/redo`，每次用 `git diff` 核对工作区。

## 用同一个只读 Agent 做前后对照

在修改前后各执行一次相同的命令。CLI 复用当前 OpenCode 服务的模型连接。需要指定模型时，加 `--model` 和 `/models` 中可用的模型 ID。

```bash
opencode run --agent task-reviewer \
  --file ranking.py --file check_ranking.py \
  "Review the winner selection and report the evidence and verdict."
```

未修复的实现通常应得到 `VERDICT: TIE_BREAK`，修复后的实现应能指出并列规则并给出 `VERDICT: OK`。手动运行的检查是验收依据。把模型判定与检查结果对照，能看出审阅 Agent 的漏报和误报。想看机器可读事件，可加 `--format json`，输出是逐行 JSON。[OpenCode CLI 命令](https://opencode.ai/docs/cli/commands/)

审阅员示例改编自 Damien Sileo，版权与 MIT 许可见示例目录的 [LICENSE.upstream](https://github.com/Haozhe-Xing/agent_learning/blob/main/examples/opencode-workshop/LICENSE.upstream)。想扩展到带标注样本的批量评估，可继续阅读[原课第 4 部分](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/README.md#4-measure-and-improve-15-min)。
