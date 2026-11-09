"""复现血糖调控作业的模型学习、Q-Learning 与报告图表。

在包含 assignment3/ 的上一级目录运行：

    python assignment3/main.py

glucose_env.py 与 sanity_check.py 放在该上一级目录，只用于本地实验，不提交。
"""

import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from glucose_env import (  # noqa: E402
    ACTION_NAMES,
    STATE_NAMES,
    GlucoseEnv,
    collect_random_episodes,
)
from model_based import ModelBasedAgent  # noqa: E402
from q_learning import QLearningAgent  # noqa: E402

FIG_DIR = ROOT / "figures"
DATA_SEED = 2026
N_TRAIN = 2000
EVAL_SEED = 50_000
N_EVAL = 500
ABLATION_SIZES = (500, 1000, 2000)
ONLINE_EPISODES = 500
ONLINE_SEEDS = (0, 1, 2, 3, 4)
ONLINE_WINDOW = 50
GAMMA = 0.9
ALPHA = 0.5

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACTION_COLOR = {
    "w": "#4C78A8",
    "g": "#F58518",
    "i1": "#54A24B",
    "i2": "#E45756",
}
ACTION_LABEL = {
    "w": "w 观察",
    "g": "g 补糖",
    "i1": "i1 小剂量",
    "i2": "i2 大剂量",
}


def recount(episodes):
    counts = {
        state: {action: {next_state: 0 for next_state in STATE_NAMES} for action in ACTION_NAMES}
        for state in STATE_NAMES
    }
    reward_sum = {
        state: {action: 0.0 for action in ACTION_NAMES} for state in STATE_NAMES
    }
    visits = {state: {action: 0 for action in ACTION_NAMES} for state in STATE_NAMES}
    for episode in episodes:
        for state, action, reward, next_state, _done in episode:
            counts[state][action][next_state] += 1
            reward_sum[state][action] += float(reward)
            visits[state][action] += 1
    return counts, reward_sum, visits


def self_check(episodes):
    counts, reward_sum, visits = recount(episodes)
    agent = ModelBasedAgent(states=list(STATE_NAMES), actions=list(ACTION_NAMES), gamma=GAMMA)
    agent.learn_model(episodes)
    for state in STATE_NAMES:
        for action in ACTION_NAMES:
            dist = agent.get_transition(state, action)
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-9:
                raise AssertionError(f"P({state},{action}) sums to {total}")
            n = visits[state][action]
            if n == 0:
                if agent.get_reward(state, action) != 0.0:
                    raise AssertionError("unseen reward must be 0")
                if any(abs(dist[sp] - 1.0 / len(STATE_NAMES)) > 1e-12 for sp in STATE_NAMES):
                    raise AssertionError("unseen transition must be uniform")
            else:
                expected_r = reward_sum[state][action] / n
                if abs(agent.get_reward(state, action) - expected_r) > 1e-9:
                    raise AssertionError("reward mean mismatch")
                for next_state in STATE_NAMES:
                    expected_p = counts[state][action][next_state] / n
                    if abs(dist[next_state] - expected_p) > 1e-9:
                        raise AssertionError("transition count mismatch")

    agent.run_value_iteration()
    residual = 0.0
    for state in STATE_NAMES:
        best = max(
            agent.get_reward(state, action)
            + GAMMA * sum(
                prob * agent.get_value(next_state)
                for next_state, prob in agent.get_transition(state, action).items()
            )
            for action in ACTION_NAMES
        )
        residual = max(residual, abs(best - agent.get_value(state)))
    if residual >= 1e-4:
        raise AssertionError(f"value iteration residual {residual}")

    state, action, reward, next_state, done = episodes[0][0]
    q_agent = QLearningAgent(
        states=list(STATE_NAMES), actions=list(ACTION_NAMES), gamma=GAMMA, alpha=ALPHA
    )
    q_agent.update(state, action, reward, next_state, bool(done))
    target = float(reward) if done else float(reward)
    expected = (1.0 - ALPHA) * 0.0 + ALPHA * target
    if abs(q_agent.get_q_value(state, action) - expected) > 1e-12:
        raise AssertionError("first Q update mismatch")

    follower = QLearningAgent(
        states=list(STATE_NAMES), actions=list(ACTION_NAMES), gamma=GAMMA, alpha=0.5
    )
    follower._Q["B3"]["w"] = 2.0
    follower.update("B4", "i1", 0.0, "B3", False)
    # (1-0.5)*0 + 0.5*(0 + 0.9*2) = 0.9
    if abs(follower.get_q_value("B4", "i1") - 0.9) > 1e-12:
        raise AssertionError("bootstrap update mismatch")
    follower.update("B4", "i1", -1.0, "B1", True)
    # (1-0.5)*0.9 + 0.5*(-1) = -0.05
    if abs(follower.get_q_value("B4", "i1") + 0.05) > 1e-12:
        raise AssertionError("terminal update mismatch")
    if follower.get_policy("B4") != "w":
        raise AssertionError("tie-break should keep the earlier action when Q stays below 0")


def evaluate(policy_fn, n_episodes=N_EVAL, seed=EVAL_SEED, random_actions=False):
    returns = []
    lengths = []
    b3_steps = 0
    total_steps = 0
    b1_ends = 0
    action_rng = random.Random(seed + 17)
    for index in range(n_episodes):
        env = GlucoseEnv(seed=seed + index)
        state = env.reset()
        done = False
        total = 0.0
        steps = 0
        while not done:
            if state == "B3":
                b3_steps += 1
            total_steps += 1
            if random_actions:
                action = action_rng.choice(ACTION_NAMES)
            else:
                action = policy_fn(state)
            state, reward, done, _info = env.step(action)
            total += reward
            steps += 1
        returns.append(total)
        lengths.append(steps)
        if state == "B1":
            b1_ends += 1
    arr = np.asarray(returns, dtype=float)
    return {
        "mean_return": float(arr.mean()),
        "std_return": float(arr.std(ddof=1)),
        "sem_return": float(arr.std(ddof=1) / np.sqrt(n_episodes)),
        "b3_ratio": b3_steps / total_steps,
        "b1_rate": b1_ends / n_episodes,
        "mean_length": float(np.mean(lengths)),
        "returns": arr.tolist(),
    }


def train_online(seed, decay, n_episodes=ONLINE_EPISODES):
    agent = QLearningAgent(
        states=list(STATE_NAMES),
        actions=list(ACTION_NAMES),
        gamma=GAMMA,
        alpha=ALPHA,
        epsilon=0.3,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        seed=seed,
    )
    rng = random.Random(10_000 + seed)
    epsilon = agent.epsilon
    returns = []
    for episode in range(n_episodes):
        env = GlucoseEnv(seed=1_000_000 + seed * 10_000 + episode)
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            if rng.random() < epsilon:
                action = rng.choice(ACTION_NAMES)
            else:
                action = agent.get_policy(state)
            next_state, reward, done, _info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total += reward
        returns.append(total)
        if decay:
            epsilon = max(agent.epsilon_min, epsilon * agent.epsilon_decay)
    return agent, returns, epsilon


def trailing_mean(values, window):
    totals = np.cumsum(values, dtype=float)
    out = np.empty(len(values), dtype=float)
    for index in range(len(values)):
        start = index - window + 1
        if start <= 0:
            out[index] = totals[index] / (index + 1)
        else:
            out[index] = (totals[index] - totals[start - 1]) / window
    return out


def fit_model_based(episodes):
    agent = ModelBasedAgent(states=list(STATE_NAMES), actions=list(ACTION_NAMES), gamma=GAMMA)
    agent.learn_model(episodes)
    agent.run_value_iteration()
    return agent


def fit_q(episodes):
    agent = QLearningAgent(
        states=list(STATE_NAMES),
        actions=list(ACTION_NAMES),
        gamma=GAMMA,
        alpha=ALPHA,
    )
    agent.learn(episodes, mode="fixed")
    return agent


def policy_snapshot(mb_agent, q_agent):
    rows = []
    for state in STATE_NAMES:
        q_values = {action: q_agent.get_q_value(state, action) for action in ACTION_NAMES}
        mb_values = {}
        for action in ACTION_NAMES:
            expected = sum(
                prob * mb_agent.get_value(next_state)
                for next_state, prob in mb_agent.get_transition(state, action).items()
            )
            mb_values[action] = mb_agent.get_reward(state, action) + GAMMA * expected
        rows.append({
            "state": state,
            "mb_action": mb_agent.get_policy(state),
            "mb_value": mb_agent.get_value(state),
            "mb_q": mb_values,
            "q_action": q_agent.get_policy(state),
            "q_values": q_values,
        })
    nonterminal = [row for row in rows if row["state"] != "B1"]
    agree = sum(row["mb_action"] == row["q_action"] for row in nonterminal)
    return rows, agree / len(nonterminal)


def unseen_pairs(visits):
    missing = []
    for state in STATE_NAMES:
        for action in ACTION_NAMES:
            if visits[state][action] == 0:
                missing.append(f"{state},{action}")
    return missing


def summarize_metrics(metrics):
    return {
        key: metrics[key]
        for key in ("mean_return", "std_return", "sem_return", "b3_ratio", "b1_rate", "mean_length")
    }


def plot_visits(visits, path):
    x = np.arange(len(STATE_NAMES))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    for index, action in enumerate(ACTION_NAMES):
        heights = [visits[state][action] for state in STATE_NAMES]
        ax.bar(
            x + (index - 1.5) * width,
            heights,
            width,
            label=ACTION_LABEL[action],
            color=ACTION_COLOR[action],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("访问次数 N(s, a)")
    ax.set_xlabel("状态")
    ax.set_title("随机策略数据中每个 (状态, 动作) 的访问次数")
    ax.legend(ncol=4, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _label_bars(ax, heights, labels):
    span = max(max(heights) - min(heights), 1.0)
    pad = span * 0.1
    for index, (height, label) in enumerate(zip(heights, labels)):
        if height >= 0:
            ax.text(index, height, label, ha="center", va="bottom", fontsize=9)
        else:
            ax.text(index, height, label, ha="center", va="top", fontsize=9)
    lower = min(0.0, min(heights)) - pad
    upper = max(0.0, max(heights)) + pad
    ax.set_ylim(lower, upper)


def plot_values(rows, path):
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    mb_heights = [row["mb_value"] for row in rows]
    mb_colors = [ACTION_COLOR[row["mb_action"]] for row in rows]
    axes[0].bar(x, mb_heights, color=mb_colors)
    _label_bars(axes[0], mb_heights, [row["mb_action"] for row in rows])
    axes[0].axhline(0.0, color="#666666", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([row["state"] for row in rows])
    axes[0].set_ylabel("V(s)")
    axes[0].set_title("模型路线的状态价值与策略")

    q_heights = [max(row["q_values"].values()) for row in rows]
    q_colors = [ACTION_COLOR[row["q_action"]] for row in rows]
    axes[1].bar(x, q_heights, color=q_colors)
    _label_bars(axes[1], q_heights, [row["q_action"] for row in rows])
    axes[1].axhline(0.0, color="#666666", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([row["state"] for row in rows])
    axes[1].set_ylabel("max Q(s, a)")
    axes[1].set_title("Q-Learning 的贪心动作")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_q_table(q_agent, path):
    table = np.array([
        [q_agent.get_q_value(state, action) for action in ACTION_NAMES]
        for state in STATE_NAMES
    ])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    image = ax.imshow(table, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES)
    ax.set_yticks(np.arange(len(STATE_NAMES)))
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel("动作")
    ax.set_title("离线 Q-Learning 的 Q 表")
    for row in range(table.shape[0]):
        for col in range(table.shape[1]):
            ax.text(col, row, f"{table[row, col]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ablation(ablation, path):
    labels = ["平均回报", "B3 时间比例", "进入 B1 的比例"]
    keys = ["mean_return", "b3_ratio", "b1_rate"]
    sizes = list(ABLATION_SIZES)
    x = np.arange(len(sizes))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for ax, label, key in zip(axes, labels, keys):
        mb = [ablation[str(size)]["model_based"][key] for size in sizes]
        ql = [ablation[str(size)]["q_learning"][key] for size in sizes]
        ax.bar(x - width / 2, mb, width, label="学模型再规划", color="#4C78A8")
        ax.bar(x + width / 2, ql, width, label="Q-Learning", color="#E45756")
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.set_xlabel("使用的 episode 数")
        ax.set_title(label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.12),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(comparison, path):
    names = ["随机策略", "学模型再规划", "Q-Learning"]
    keys_src = ["random", "model_based", "q_learning"]
    colors = ["#9D9D9D", "#4C78A8", "#E45756"]
    specs = [
        ("mean_return", "平均无折扣回报"),
        ("b3_ratio", "B3 时间比例"),
        ("b1_rate", "进入 B1 的比例"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    for ax, (key, title) in zip(axes, specs):
        vals = [comparison[name][key] for name in keys_src]
        errors = [comparison[name]["sem_return"] if key == "mean_return" else 0.0 for name in keys_src]
        ax.bar(np.arange(3), vals, color=colors, yerr=errors, capsize=3)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(names, rotation=15)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_online(curves, path):
    episodes = np.arange(1, ONLINE_EPISODES + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    styles = {
        "fixed": ("#4C78A8", "固定 ε = 0.3"),
        "decay": ("#F58518", "衰减 ε：0.3 × 0.995，下限 0.05"),
    }
    for key, (color, label) in styles.items():
        smoothed = np.vstack([
            trailing_mean(np.asarray(seed_returns, dtype=float), ONLINE_WINDOW)
            for seed_returns in curves[key]
        ])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0, ddof=1)
        ax.plot(episodes, mean, color=color, label=label, linewidth=2)
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel("在线 episode")
    ax.set_ylabel("无折扣回报（窗口 50）")
    ax.set_title("在线 Q-Learning：5 个种子的学习曲线")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def dataset_summary(episodes, visits):
    lengths = [len(episode) for episode in episodes]
    b1_ends = sum(episode[-1][3] == "B1" for episode in episodes)
    flat_counts = [visits[state][action] for state in STATE_NAMES for action in ACTION_NAMES]
    return {
        "n_episodes": len(episodes),
        "n_transitions": int(sum(lengths)),
        "mean_length": float(np.mean(lengths)),
        "b1_end_rate": b1_ends / len(episodes),
        "min_visits": int(min(flat_counts)),
        "max_visits": int(max(flat_counts)),
        "unseen": unseen_pairs(visits),
    }


BIN_LOW = {name: 2.0 + 2.0 * index for index, name in enumerate(STATE_NAMES)}


class VisitAverageQ(QLearningAgent):
    """步长 α = 1/N(s,a)。只作为报告里的对照，提交的主算法仍用固定 α。"""

    def __init__(self, states, actions, gamma=GAMMA):
        super().__init__(states=states, actions=actions, gamma=gamma, alpha=1.0)
        self._visits = {
            state: {action: 0 for action in self.actions} for state in self.states
        }

    def update(self, state, action, reward, next_state, done):
        self._visits[state][action] += 1
        self.alpha = 1.0 / self._visits[state][action]
        super().update(state, action, reward, next_state, bool(done))


def fit_visit_average_q(episodes):
    agent = VisitAverageQ(states=list(STATE_NAMES), actions=list(ACTION_NAMES))
    agent.learn(episodes, mode="fixed")
    return agent


def ranked_pair(qmap):
    best_action = ACTION_NAMES[0]
    best_q = qmap[best_action]
    second_action = None
    second_q = None
    for action in ACTION_NAMES[1:]:
        q = qmap[action]
        if q > best_q:
            second_action, second_q = best_action, best_q
            best_action, best_q = action, q
        elif second_action is None or q > second_q:
            second_action, second_q = action, q
    return {
        "best": best_action,
        "second": second_action,
        "gap": best_q - second_q,
    }


def rollout_details(policy_fn, n_episodes, seed):
    """执行策略时记录决策时刻的真实血糖，只用于评估，不写回算法。"""
    next_counts = {state: {nxt: 0 for nxt in STATE_NAMES} for state in STATE_NAMES}
    half_counts = {
        state: [
            {nxt: 0 for nxt in STATE_NAMES},
            {nxt: 0 for nxt in STATE_NAMES},
        ]
        for state in STATE_NAMES
    }
    for index in range(n_episodes):
        env = GlucoseEnv(seed=seed + index)
        state = env.reset()
        done = False
        while not done:
            glucose = env.get_true_glucose()
            action = policy_fn(state)
            next_state, _reward, done, _info = env.step(action)
            fraction = (glucose - BIN_LOW[state]) / 2.0
            half = 0 if fraction < 0.5 else 1
            next_counts[state][next_state] += 1
            half_counts[state][half][next_state] += 1
            state = next_state
    return next_counts, half_counts


def distribution(counts):
    total = sum(counts.values())
    if total == 0:
        return {state: 0.0 for state in STATE_NAMES}, 0
    return {state: counts[state] / total for state in STATE_NAMES}, total


def total_variation(left, right):
    return 0.5 * sum(abs(left[state] - right[state]) for state in STATE_NAMES)


def plot_action_gaps(gaps, path):
    x = np.arange(len(STATE_NAMES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.bar(x - width / 2, [gaps[s]["model"]["gap"] for s in STATE_NAMES], width,
           label="计数模型", color="#4C78A8")
    ax.bar(x + width / 2, [gaps[s]["q"]["gap"] for s in STATE_NAMES], width,
           label="固定 α = 0.5", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES)
    ax.set_ylabel("最优动作与次优动作的 Q 差")
    ax.set_title("每个状态上的动作间隙")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_within_bin(halves, path):
    focus = ["B3", "B4", "B5"]
    show = ["B1", "B2", "B3", "B4", "B5", "B6"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.9), sharey=True)
    x = np.arange(len(show))
    width = 0.36
    for ax, state in zip(axes, focus):
        low, _n_low = distribution(halves[state][0])
        high, _n_high = distribution(halves[state][1])
        ax.bar(x - width / 2, [low[s] for s in show], width, label="区间下半", color="#4C78A8")
        ax.bar(x + width / 2, [high[s] for s in show], width, label="区间上半", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(show, rotation=0)
        ax.set_title(state)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("下一状态比例")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("同一离散状态里，区间下半和上半的下一状态", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_model_shift(shift_rows, path):
    names = [row["state"] for row in shift_rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x, [row["tv"] for row in shift_rows], color="#4C78A8")
    for index, row in enumerate(shift_rows):
        ax.text(index, row["tv"], str(row["n"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("总变差距离")
    ax.set_xlabel("状态（柱顶是执行策略下的样本数）")
    ax.set_title("随机数据上学到的转移，和执行策略时的经验转移")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def innovation_study(episodes, mb_agent, rows):
    gaps = {}
    for row in rows:
        gaps[row["state"]] = {
            "model": ranked_pair(row["mb_q"]),
            "q": ranked_pair(row["q_values"]),
        }
    plot_action_gaps(gaps, FIG_DIR / "action_gap.png")

    decreasing = {}
    for size in ABLATION_SIZES:
        agent = fit_visit_average_q(episodes[:size])
        decreasing[str(size)] = {state: agent.get_policy(state) for state in STATE_NAMES}

    next_counts, half_counts = rollout_details(mb_agent.get_policy, n_episodes=500, seed=90_000)
    shift = []
    half_summary = {}
    for state in STATE_NAMES:
        empirical, n = distribution(next_counts[state])
        action = mb_agent.get_policy(state)
        model = mb_agent.get_transition(state, action)
        low, n_low = distribution(half_counts[state][0])
        high, n_high = distribution(half_counts[state][1])
        shift.append({
            "state": state,
            "action": action,
            "n": n,
            "tv": total_variation(empirical, model) if n else None,
            "empirical_top": sorted(empirical.items(), key=lambda item: -item[1])[:3],
        })
        half_summary[state] = {
            "n_low": n_low,
            "n_high": n_high,
            "low": {s: low[s] for s in STATE_NAMES if low[s] >= 0.02 or s == state},
            "high": {s: high[s] for s in STATE_NAMES if high[s] >= 0.02 or s == state},
        }
    usable = [row for row in shift if row["n"] >= 30 and row["tv"] is not None]
    plot_model_shift(usable, FIG_DIR / "model_shift.png")
    plot_within_bin(half_counts, FIG_DIR / "within_bin.png")
    return {
        "action_gap": gaps,
        "decreasing_alpha_policy": decreasing,
        "model_shift": shift,
        "within_bin": half_summary,
    }


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("收集 2000 条随机 episode ...")
    episodes = collect_random_episodes(n_episodes=N_TRAIN, seed=DATA_SEED)
    self_check(episodes)
    _counts, _reward_sum, visits = recount(episodes)
    summary = dataset_summary(episodes, visits)

    print("训练两条离线路线 ...")
    mb_agent = fit_model_based(episodes)
    q_agent = fit_q(episodes)
    rows, agreement = policy_snapshot(mb_agent, q_agent)

    print("评估 500 条轨迹 ...")
    comparison = {
        "random": evaluate(None, random_actions=True),
        "model_based": evaluate(mb_agent.get_policy),
        "q_learning": evaluate(q_agent.get_policy),
    }
    paired = np.asarray(comparison["model_based"]["returns"]) - np.asarray(comparison["q_learning"]["returns"])

    print("数据量消融 ...")
    ablation = {}
    for size in ABLATION_SIZES:
        subset = episodes[:size]
        mb_sub = fit_model_based(subset)
        q_sub = fit_q(subset)
        _c, _r, sub_visits = recount(subset)
        sub_rows, sub_agree = policy_snapshot(mb_sub, q_sub)
        ablation[str(size)] = {
            "model_based": summarize_metrics(evaluate(mb_sub.get_policy)),
            "q_learning": summarize_metrics(evaluate(q_sub.get_policy)),
            "unseen": unseen_pairs(sub_visits),
            "agreement": sub_agree,
            "mb_policy": {row["state"]: row["mb_action"] for row in sub_rows},
            "q_policy": {row["state"]: row["q_action"] for row in sub_rows},
            "mb_iterations": mb_sub.iterations_,
            "mb_delta": mb_sub.delta_,
        }

    print("在线探索实验 ...")
    online_curves = {"fixed": [], "decay": []}
    online_final = {"fixed": [], "decay": []}
    final_eps = {"fixed": [], "decay": []}
    for seed in ONLINE_SEEDS:
        for key, decay in (("fixed", False), ("decay", True)):
            agent, returns, epsilon = train_online(seed, decay=decay)
            online_curves[key].append(returns)
            final_eps[key].append(epsilon)
            online_final[key].append(summarize_metrics(evaluate(agent.get_policy, n_episodes=200, seed=70_000)))
            print(f"  seed {seed} {key} 最后 ε={epsilon:.3f} 末段回报={np.mean(returns[-100:]):.2f}")

    def online_block(key):
        raw = np.asarray(online_curves[key], dtype=float)
        tail = raw[:, -100:].mean(axis=1)
        greedy = online_final[key]
        return {
            "tail_mean_return": float(tail.mean()),
            "tail_std_across_seeds": float(tail.std(ddof=1)),
            "final_epsilon": final_eps[key],
            "greedy_mean_return": float(np.mean([item["mean_return"] for item in greedy])),
            "greedy_b3_ratio": float(np.mean([item["b3_ratio"] for item in greedy])),
            "greedy_b1_rate": float(np.mean([item["b1_rate"] for item in greedy])),
        }

    print("动作间隙、步长对照和执行时的转移 ...")
    extra = innovation_study(episodes, mb_agent, rows)

    reward_table = {
        state: {action: mb_agent.get_reward(state, action) for action in ACTION_NAMES}
        for state in STATE_NAMES
    }
    visit_table = {
        state: {action: visits[state][action] for action in ACTION_NAMES}
        for state in STATE_NAMES
    }

    results = {
        "data_seed": DATA_SEED,
        "eval_seed": EVAL_SEED,
        "n_eval": N_EVAL,
        "gamma": GAMMA,
        "alpha": ALPHA,
        "dataset": summary,
        "visit_counts": visit_table,
        "reward_hat": reward_table,
        "value_iterations": mb_agent.iterations_,
        "value_delta": mb_agent.delta_,
        "agreement_nonterminal": agreement,
        "policies": [
            {
                "state": row["state"],
                "mb_action": row["mb_action"],
                "mb_value": row["mb_value"],
                "mb_q": row["mb_q"],
                "q_action": row["q_action"],
                "q_values": row["q_values"],
            }
            for row in rows
        ],
        "comparison": {name: summarize_metrics(metrics) for name, metrics in comparison.items()},
        "paired_mb_minus_q": {
            "mean": float(paired.mean()),
            "sem": float(paired.std(ddof=1) / np.sqrt(len(paired))),
        },
        "ablation": ablation,
        "online": {"fixed": online_block("fixed"), "decay": online_block("decay")},
        "innovation": extra,
    }

    def convert(value):
        if isinstance(value, float):
            return round(value, 6)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    (ROOT / "results.json").write_text(
        json.dumps(convert(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    plot_visits(visits, FIG_DIR / "visit_counts.png")
    plot_values(rows, FIG_DIR / "values_and_policy.png")
    plot_q_table(q_agent, FIG_DIR / "q_table.png")
    plot_ablation(ablation, FIG_DIR / "ablation.png")
    plot_metrics(comparison, FIG_DIR / "metrics_compare.png")
    plot_online(online_curves, FIG_DIR / "online_learning.png")

    print("\n策略与价值")
    for row in rows:
        print(
            f"  {row['state']}: MB {row['mb_action']} V={row['mb_value']:.3f}"
            f" | QL {row['q_action']} maxQ={max(row['q_values'].values()):.3f}"
        )
    print(f"非终止状态一致率 {agreement:.3f}")
    print(f"价值迭代 {mb_agent.iterations_} 轮, delta={mb_agent.delta_:.3e}")
    for name, metrics in comparison.items():
        print(
            f"{name}: return={metrics['mean_return']:.3f}±{metrics['sem_return']:.3f}"
            f"  B3={metrics['b3_ratio']:.3f}  B1={metrics['b1_rate']:.3f}"
            f"  len={metrics['mean_length']:.1f}"
        )
    print("结果写入 assignment3/results.json 与 assignment3/figures/")


if __name__ == "__main__":
    main()
