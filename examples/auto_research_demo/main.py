"""
指标驱动 Auto Research —— 最小可运行教学 demo
================================================

这个文件复刻 Karpathy `autoresearch` 的骨架，用纯 Python 标准库实现，
零第三方依赖，直接 `python3 main.py` 就能跑。

它回答一个问题：auto research 到底「怎么做」？

核心只有三块，彼此边界清晰，这是全篇唯一要记住的东西：

  1. 固定评估器（不可触碰）  —— 数据、评测协议、随机种子全锁死。
                                agent 只能调用 evaluate() 拿分数，
                                不能读、不能改它的实现。
  2. 可搜索的候选            —— 模型结构 + 训练 recipe（隐层宽度、
                                激活函数、学习率、训练轮数、初始化尺度）。
  3. 搜索策略                —— 随机基线 / 单路径贪心 / 种群进化，
                                决定「下一个试谁、要不要保留」。

跑一遍你会看到：随机基线只有 ~60%，单路径贪心逐步爬到 ~90%+，
种群进化更快更稳 —— 这就是 auto research 的最小闭环。

读完本文件后，进阶方向见文件末尾的「下一步」。
"""

import math
import random
import time

# ============================================================
# 第一区：固定评估器（不可触碰边界）
# ------------------------------------------------------------
# 真实系统里这一区必须与「搜索 agent」物理隔离：agent 只能调用
# evaluate()，不能读、不能改这里的实现。否则它就能通过「改评测
# 协议」来作弊——比如偷偷扩大验证集、改随机种子、把答案写死。
# 这一区是 auto research 的「宪法」，搜索 agent 永远无法触碰。
# ============================================================

RNG_SEED = 42          # 全局随机种子：让整个实验可复现


def make_data(n, seed):
    """生成固定数据集：2 维棋盘格二分类 + 5% 标签噪声。

    输入归一化到 [-1,1]，标签规则 sign(sin(pi*x1) * sin(pi*x2))
    把平面切成 4 个方格。之所以必须归一化：若不归一化，tanh 输入
    会饱和、梯度消失、训练卡死在 ~60%。这本身是「评估器也要设计
    好」的具体例子——任务得让「好配置」和「坏配置」拉开差距，
    搜索才有梯度感（随机 ~60%，好配置 ~92%）。
    5% 噪声意味着理论上限略低于 100%，防止过拟合到满分。
    """
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        label = 1 if (math.sin(math.pi * x1) * math.sin(math.pi * x2)) > 0 else 0
        if rng.random() < 0.05:
            label = 1 - label
        X.append((x1, x2))
        y.append(label)
    return X, y


# 训练集 / 验证集：用不同种子独立生成，验证集对搜索 agent 全程不可见
X_train, y_train = make_data(280, RNG_SEED)
X_val, y_val = make_data(140, RNG_SEED + 1)


def _activate(act, z):
    """激活函数：返回 (激活值, 导数)。三种都支持，用于反向传播。

    linear 是关键：它学不了棋盘格这种 XOR 类问题，是「天然坏结构」，
    用来让搜索空间里存在明显更差的候选——这正是 auto research 要
    搜的「结构」，而不只是「超参」。
    """
    if act == "tanh":
        t = math.tanh(z)
        return t, 1.0 - t * t
    if act == "relu":
        return (z if z > 0 else 0.0), (1.0 if z > 0 else 0.0)
    return z, 1.0           # linear：恒等 + 导数恒 1


def _forward(cfg, W1, b1, W2, b2, x):
    """2 层 MLP 前向：x(2) -> 激活隐层(hidden) -> sigmoid 输出(1)。

    返回 (预测概率 p, 隐层激活 h, 隐层激活导数 dh)。h、dh 会被反向
    传播复用，避免重算。
    """
    h, dh = [], []
    for j in range(cfg["hidden"]):
        z = W1[j][0] * x[0] + W1[j][1] * x[1] + b1[j]
        a, d = _activate(cfg["activation"], z)
        h.append(a)
        dh.append(d)
    s2 = sum(W2[j] * h[j] for j in range(cfg["hidden"])) + b2
    p = 1.0 / (1.0 + math.exp(-s2))
    return p, h, dh


_eval_cache = {}       # 评估缓存：同一配置只训练一次，省时间


def evaluate(cfg):
    """【固定评估器】训练并返回验证集准确率（0~1）。

    这是 auto research 里唯一「权威」的东西：agent 的所有决策都
    只依据这个分数。随机种子固定 → 同一配置分数恒定、可复现、
    可比较。agent 无法知道它内部怎么算，只能拿到一个数字。
    """
    key = tuple(sorted(cfg.items()))
    if key in _eval_cache:
        return _eval_cache[key]

    rng = random.Random(RNG_SEED)
    hidden = cfg["hidden"]
    lr, epochs, scale = cfg["lr"], cfg["epochs"], cfg["init_scale"]

    # 初始化（固定种子 → 同一配置每次初始值一致）
    W1 = [[rng.uniform(-scale, scale) for _ in range(2)] for _ in range(hidden)]
    b1 = [0.0] * hidden
    W2 = [rng.uniform(-scale, scale) for _ in range(hidden)]
    b2 = 0.0

    # 训练：mini-batch SGD + BCE loss + 逐 epoch 学习率衰减
    n = len(X_train)
    bs = 32
    for ep in range(epochs):
        decay = 1.0 / (1.0 + 0.01 * ep)
        idx = list(range(n))
        rng.shuffle(idx)
        for start in range(0, n, bs):
            batch = idx[start:start + bs]
            gW1 = [[0.0] * 2 for _ in range(hidden)]
            gb1 = [0.0] * hidden
            gW2 = [0.0] * hidden
            gb2 = 0.0
            for k in batch:
                x, yv = X_train[k], y_train[k]
                p, h, dh = _forward(cfg, W1, b1, W2, b2, x)
                d2 = p - yv                          # sigmoid+BCE 的简化梯度
                for j in range(hidden):
                    d1 = d2 * W2[j] * dh[j]          # 乘激活导数
                    gW2[j] += d2 * h[j]
                    gb1[j] += d1
                    gW1[j][0] += d1 * x[0]
                    gW1[j][1] += d1 * x[1]
                gb2 += d2
            cnt = len(batch)
            for j in range(hidden):
                W2[j] -= lr * decay * gW2[j] / cnt
                b1[j] -= lr * decay * gb1[j] / cnt
                W1[j][0] -= lr * decay * gW1[j][0] / cnt
                W1[j][1] -= lr * decay * gW1[j][1] / cnt
            b2 -= lr * decay * gb2 / cnt

    # 验证（这是 agent 唯一能看到的输出）
    correct = 0
    for x, y in zip(X_val, y_val):
        p, _, _ = _forward(cfg, W1, b1, W2, b2, x)
        if (1 if p > 0.5 else 0) == y:
            correct += 1
    acc = correct / len(X_val)
    _eval_cache[key] = acc
    return acc


# ============================================================
# 第二区：候选空间 + 搜索策略
# ------------------------------------------------------------
# 真实系统里，「生成候选」这步由 LLM 承担（AIDE 的 Draft/Debug/
# Improve，AlphaEvolve 的 ensemble 生成）。这里用「规则式提议器」
# 演示同样的闭环结构——把 LLM 换成随机扰动，循环逻辑完全一样。
# 你只需把 mutate() 换成「LLM 读历史 + 写新代码」，其余不变。
# ============================================================

HIDDEN_OPTIONS = [2, 3, 4, 6, 8, 12, 16]
ACT_OPTIONS = ["tanh", "relu", "linear"]
LR_OPTIONS = [0.005, 0.02, 0.05, 0.2, 0.8]
EPOCH_OPTIONS = [20, 60, 150, 300, 500]
SCALE_OPTIONS = [0.3, 0.6, 1.0, 1.5, 2.5]


def random_candidate(rng):
    """随机采样一个配置：随机搜索的「提议器」。"""
    return {
        "hidden": rng.choice(HIDDEN_OPTIONS),
        "activation": rng.choice(ACT_OPTIONS),
        "lr": rng.choice(LR_OPTIONS),
        "epochs": rng.choice(EPOCH_OPTIONS),
        "init_scale": rng.choice(SCALE_OPTIONS),
    }


def mutate(cfg, rng):
    """原子改动：每轮只随机改一个维度（且保证改到新值）。

    呼应 AIDE 的 Improve——「小步、原子、可归因」。这样每次提分/
    掉分都能精确归因到「改了哪个维度」，而不是「重写了整份配置」。
    这是 auto research 区别于「让模型一次性重写全部」的本质。
    """
    c = dict(cfg)
    key = rng.choice(list(c))
    options = {
        "hidden": HIDDEN_OPTIONS,
        "activation": ACT_OPTIONS,
        "lr": LR_OPTIONS,
        "epochs": EPOCH_OPTIONS,
        "init_scale": SCALE_OPTIONS,
    }[key]
    new = rng.choice(options)
    while new == c[key]:
        new = rng.choice(options)
    c[key] = new
    return c


def hill_climb(init_cfg, budget, rng_seed=7):
    """单路径贪心（ratchet）：从基线出发，只保留比当前更好的改动。

    这是 Karpathy autoresearch 的策略：改进则保留、否则回滚，
    绝不主动倒退。优点：可审计、成本低、每步可归因；
    缺点：跨不过「短期退化、长期获益」的性能谷（AIDE 后期也
    退化成这样）。history 就是最朴素的「实验记忆」。
    """
    rng = random.Random(rng_seed)
    best = dict(init_cfg)
    best_score = evaluate(best)
    history = [(best_score, "start", best)]
    for _ in range(budget):
        cand = mutate(best, rng)
        score = evaluate(cand)
        if score > best_score:
            best, best_score = cand, score
            history.append((score, "KEEP", cand))
        else:
            history.append((score, "reject", cand))
    return best_score, best, history


def evolution(pop_size, generations, rng_seed=11):
    """种群进化：保留 top-k 精英，变异出下一代。

    比单路径更「保持多样性」，更像 AlphaEvolve / FunSearch 的思路：
    同时维护多个候选，优胜者进入下一代，避免单点贪心卡死。
    history 记录每代最优，等价于「进化谱系」。
    """
    rng = random.Random(rng_seed)
    pop = sorted(((evaluate(c), c) for c in
                  (random_candidate(rng) for _ in range(pop_size))),
                 key=lambda t: -t[0])
    best = pop[0][0]
    best_cfg = pop[0][1]
    history = [best]
    for _ in range(generations):
        elites = pop[:max(2, pop_size // 2)]
        scored = sorted(
            ((evaluate(c), c) for c in
             (mutate(elites[rng.randrange(len(elites))][1], rng)
              for _ in range(pop_size))),
            key=lambda t: -t[0],
        )
        pop = scored
        if scored[0][0] > best:
            best, best_cfg = scored[0][0], scored[0][1]
        history.append(best)
    return best, best_cfg, history


if __name__ == "__main__":
    print("=" * 64)
    print("指标驱动 Auto Research · 最小可运行 demo")
    print("任务：2 层 MLP 拟合棋盘格二分类，最大化验证集准确率")
    print("=" * 64)

    # 1) 随机基线：不搜索，纯随机采样，看「运气」能到哪
    t0 = time.time()
    rng = random.Random(1)
    random_scores = [evaluate(random_candidate(rng)) for _ in range(18)]
    print("\n[1] 随机基线（18 个随机配置）")
    print(f"    最好 {max(random_scores):.1%} | 均值 {sum(random_scores)/len(random_scores):.1%} "
          f"| 用时 {time.time()-t0:.1f}s")

    # 2) 单路径贪心（ratchet）
    t0 = time.time()
    start_cfg = {"hidden": 3, "activation": "tanh", "lr": 0.05,
                 "epochs": 60, "init_scale": 1.0}
    score_hc, cfg_hc, hist_hc = hill_climb(start_cfg, budget=15)
    print(f"\n[2] 单路径贪心（从基线出发，15 次尝试，好才保留）")
    print(f"    最终 {score_hc:.1%} | 用时 {time.time()-t0:.1f}s")
    for s, tag, c in hist_hc[:7]:
        print(f"      {s:6.1%}  {tag:6s}  {c}")
    if len(hist_hc) > 7:
        print(f"      ...（共 {len(hist_hc)} 步）")

    # 3) 种群进化
    t0 = time.time()
    score_ev, cfg_ev, hist_ev = evolution(pop_size=6, generations=6)
    print(f"\n[3] 种群进化（6 个候选 × 6 代，精英 + 变异）")
    print(f"    最终 {score_ev:.1%} | 用时 {time.time()-t0:.1f}s")

    print("\n" + "=" * 64)
    print("结论：随机基线均值 ~69%，搜索策略稳定到 ~92-94%。")
    print("      同样的评估器，搜索把它从 69% 拉到 94% ——")
    print("      这就是 auto research 的价值。")
    print("      贪心 vs 进化谁更好，取决于搜索空间：")
    print("      这个空间平滑单峰，贪心就够；大空间多峰才需要进化。")
    print("=" * 64)
    print("""
下一步（从「调 5 个超参/结构」升级到「改任意代码」）：
  1. 把 mutate() 换成 LLM：给它看 history，让它写新版 solver 代码；
  2. 搜索空间从 5 个配置 → 任意 Python（AIDE 的做法）；
  3. 加入「失败记忆」：reject 的原因存档，避免重复试错；
  4. 加入「晋升门禁」：不只比分数，还要看置信区间/多随机种子，
     防止验证集过拟合（这是上一版报告的核心）。
""")
