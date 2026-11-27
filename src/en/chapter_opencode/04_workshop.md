# 16A.4 Workshop: Fix a Ranking Bug

This exercise adapts Damien Sileo's MIT-licensed workshop. The original uses a tie that can make a valid answer fail, then asks learners to compare interactive changes, a read-only reviewer, and an independent check. Here the task is small enough to inspect in two Python files. [Upstream guided exercise](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/GUIDED.md) · [Book example](https://github.com/Haozhe-Xing/agent_learning/tree/main/examples/opencode-workshop)

## Make a disposable workspace

Run this from a checkout of the book repository. The commands use a macOS/Linux shell and also work in WSL. The copy includes the hidden `.opencode/` directory.

```bash
workshop_dir="$(mktemp -d)"
cp -R examples/opencode-workshop/. "$workshop_dir/"
cd "$workshop_dir"
git init
git add .
git -c user.name="OpenCode Workshop" -c user.email=workshop@example.com commit -m "Save workshop baseline"
python3 -B check_ranking.py
```

The last command should fail on a tie. The highest score wins, and for equal scores `alpha` should precede `zeta` regardless of input order. The script also checks an ordinary, non-tied case.

## Plan the fix, then edit only the implementation

1. Start `opencode`, connect a model, and select **Plan** with `/agents`. Attach the two files and ask: “Why does the tie in `@check_ranking.py` fail? Point to the cause in `@ranking.py` and propose the smallest fix.”
2. Before editing, you can run the read-only review below and record its `VERDICT`.
3. Switch to **Build** using `/agents`. Ask it to edit only `ranking.py` and keep the function signature. Inspect the file named in the permission prompt before approving the expected change.
4. Run `git diff -- ranking.py`, `git status --short`, and `python3 -B check_ranking.py` from your own terminal. Expect only `ranking.py` to change and the check to print `Ranking checks passed.` To practice recovery before committing, try `/undo` and `/redo` and inspect the diff after each.

## Run the same read-only review before and after

Run the same command once before the change and once after. The CLI uses the current OpenCode server's model connections. To pin a model, pass `--model` with an ID available in `/models`.

```bash
opencode run --agent task-reviewer \
  --file ranking.py --file check_ranking.py \
  "Review the winner selection and report the evidence and verdict."
```

Expect the reviewer to report `VERDICT: TIE_BREAK` before the fix and `VERDICT: OK` afterward, backed by evidence. Use the independent check for acceptance and compare its output with the model's verdict to spot misses and false alarms. For machine-readable events, add `--format json` to emit newline-delimited JSON. [OpenCode CLI command reference](https://opencode.ai/docs/cli/commands/)

The reviewer derives from Damien Sileo's example. Its copyright notice and MIT terms are in [LICENSE.upstream](https://github.com/Haozhe-Xing/agent_learning/blob/main/examples/opencode-workshop/LICENSE.upstream). For a larger evaluation on labelled examples, follow [part 4 of the original tutorial](https://github.com/sileod/opencode-tutorial/blob/f805c049ed40e4645058d349d5a3905f4f5a92bd/README.md#4-measure-and-improve-15-min).
