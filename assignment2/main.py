"""在 MNIST 上训练并评估朴素贝叶斯和小型贝叶斯网络。

数据默认放到仓库根目录的 assignment2-data/mnist/，不进入提交内容。
"""

import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from bayes_net import FEATURE_NAMES, FEATURE_PARENTS, BayesNetClassifier
from naive_bayes import NaiveBayesClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "assignment2-data" / "mnist"
FIG_DIR = ROOT / "figures"

MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def ensure_mnist(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, md5 in MNIST_FILES.items():
        dest = data_dir / name
        if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == md5:
            continue
        last_error = None
        for base in MIRRORS:
            try:
                urllib.request.urlretrieve(base + name, dest)
                digest = hashlib.md5(dest.read_bytes()).hexdigest()
                if digest == md5:
                    last_error = None
                    break
                last_error = f"{name} 校验失败"
            except Exception as exc:
                last_error = str(exc)
        if last_error:
            raise RuntimeError(f"下载 {name} 失败：{last_error}")


def load_images(path):
    with gzip.open(path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols)


def load_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64).copy()


def load_mnist(data_dir: Path):
    x_train = load_images(data_dir / "train-images-idx3-ubyte.gz")
    y_train = load_labels(data_dir / "train-labels-idx1-ubyte.gz")
    x_test = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
    y_test = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def confusion_matrix(y_true, y_pred, n_classes=10):
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def scores_from_cm(matrix):
    total = matrix.sum()
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    precision = []
    recall = []
    f1 = []
    for c in range(matrix.shape[0]):
        tp = float(matrix[c, c])
        fp = float(matrix[:, c].sum() - tp)
        fn = float(matrix[c, :].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "confusion_matrix": matrix.tolist(),
    }


def top_confusions(matrix, k=6):
    pairs = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                pairs.append((int(matrix[i, j]), i, j))
    pairs.sort(reverse=True)
    return [{"count": count, "true": i, "pred": j} for count, i, j in pairs[:k]]


def pick_error_indices(y_true, y_pred, proba, pairs, n_images=6):
    chosen = []
    for pair in pairs:
        candidates = np.flatnonzero((y_true == pair["true"]) & (y_pred == pair["pred"]))
        if candidates.size == 0:
            continue
        order = np.argsort(-proba[candidates, pair["pred"]])
        for index in candidates[order[:2]]:
            chosen.append(int(index))
            if len(chosen) >= n_images:
                return chosen
    return chosen


def plot_confusion(matrix, path, title):
    """格内是样本数。颜色用对数刻度，避免对角线把错分格冲淡。"""
    from matplotlib.colors import LogNorm

    shown = np.ma.masked_where(matrix <= 0, matrix.astype(float))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f7f8fa")
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    image = ax.imshow(shown, cmap=cmap, norm=LogNorm(vmin=1, vmax=max(int(matrix.max()), 1)))
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.4)
    ax.tick_params(which="minor", length=0)
    darkest = max(int(matrix.max()) * 0.18, 40)
    for i in range(10):
        for j in range(10):
            count = int(matrix[i, j])
            if count == 0:
                ax.text(j, i, "0", ha="center", va="center", color="#c5cdd6", fontsize=8)
            else:
                color = "white" if count >= darkest else "#1c2833"
                ax.text(j, i, str(count), ha="center", va="center", color=color, fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("样本数（对数刻度）")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_errors(images, y_true, y_pred, indices, path, title):
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 6.8))
    for ax, index in zip(axes.ravel(), indices):
        ax.imshow(images[index].reshape(28, 28), cmap="gray_r", interpolation="nearest")
        true_label = int(y_true[index])
        pred_label = int(y_pred[index])
        ax.set_title(f"真实 {true_label}  →  预测 {pred_label}\n测试集 #{index}", fontsize=11, color="#9b2c2c", pad=6)
        ax.axis("off")
    for ax in axes.ravel()[len(indices):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(nb_scores, bn_scores, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.05, 1.35]})
    labels = ["Accuracy", "Macro P", "Macro R", "Macro F1"]
    nb_vals = [nb_scores["accuracy"], nb_scores["macro_precision"], nb_scores["macro_recall"], nb_scores["macro_f1"]]
    bn_vals = [bn_scores["accuracy"], bn_scores["macro_precision"], bn_scores["macro_recall"], bn_scores["macro_f1"]]
    x = np.arange(len(labels))
    width = 0.36
    nb_color, bn_color = "#2f5d8c", "#d4762c"
    left = axes[0]
    nb_bars = left.bar(x - width / 2, nb_vals, width, label="朴素贝叶斯", color=nb_color)
    bn_bars = left.bar(x + width / 2, bn_vals, width, label="贝叶斯网络", color=bn_color)
    left.set_ylim(0, 1.12)
    left.set_xticks(x)
    left.set_xticklabels(labels)
    left.set_ylabel("分数")
    left.set_title("总体指标")
    for bars in (nb_bars, bn_bars):
        for bar in bars:
            height = bar.get_height()
            left.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha="center", va="bottom", fontsize=8)

    right = axes[1]
    digits = np.arange(10)
    right.bar(digits - width / 2, nb_scores["per_class_f1"], width, label="朴素贝叶斯", color=nb_color)
    right.bar(digits + width / 2, bn_scores["per_class_f1"], width, label="贝叶斯网络", color=bn_color)
    right.set_ylim(0, 1.12)
    right.set_xticks(digits)
    right.set_xlabel("数字")
    right.set_ylabel("F1")
    right.set_title("各类 F1")
    handles, legend_labels = right.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_top_pairs(nb_pairs, bn_pairs, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    panels = (
        (axes[0], nb_pairs, "朴素贝叶斯", "#2f5d8c"),
        (axes[1], bn_pairs, "贝叶斯网络", "#d4762c"),
    )
    for ax, pairs, title, color in panels:
        rows = list(reversed(pairs[:6]))
        labels = [f"{item['true']} → {item['pred']}" for item in rows]
        counts = [item["count"] for item in rows]
        ax.barh(labels, counts, color=color, height=0.62)
        ceiling = max(counts) * 1.18 if counts else 1
        ax.set_xlim(0, ceiling)
        ax.set_xlabel("张数")
        ax.set_title(title)
        for y, count in enumerate(counts):
            ax.text(count + ceiling * 0.02, y, str(count), va="center", fontsize=9)
    fig.suptitle("最多的错分对", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_structure(path):
    """手绘 DAG。灰箭头从类别指出，橙箭头是特征之间的依赖。"""
    labels = {
        "Y": "类别 Y",
        "top": "上墨",
        "mid": "中墨",
        "bot": "下墨",
        "left": "左墨",
        "right": "右墨",
        "aspect": "宽高比",
        "h_sym": "左右对称",
        "v_sym": "上下对称",
        "holes": "孔数",
        "hole_pos": "孔位置",
        "cross": "中线穿越",
        "profile": "上下宽差",
    }
    pos = {
        "Y": (5.0, 3.55),
        "top": (0.2, 2.15),
        "left": (2.5, 2.15),
        "aspect": (4.8, 2.15),
        "holes": (7.1, 2.15),
        "profile": (9.4, 2.15),
        "mid": (0.2, 0.85),
        "v_sym": (1.7, 0.85),
        "right": (3.5, 0.85),
        "h_sym": (5.3, 0.85),
        "hole_pos": (7.6, 0.85),
        "bot": (0.2, -0.45),
        "cross": (2.2, -0.45),
    }
    feature_edges = [
        ("top", "mid"),
        ("mid", "bot"),
        ("left", "right"),
        ("left", "h_sym"),
        ("top", "v_sym"),
        ("holes", "hole_pos"),
        ("mid", "cross"),
    ]
    children = {name for _, name in feature_edges}
    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    ax.set_xlim(-0.7, 10.8)
    ax.set_ylim(-1.15, 4.25)
    ax.axis("off")
    ax.set_title("小型贝叶斯网络结构", fontsize=14, pad=8)

    def arrow(a, b, color, rad):
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        patch = FancyArrowPatch(
            (x1, y1 - 0.16),
            (x2, y2 + 0.22),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=8,
            shrinkB=2,
        )
        ax.add_patch(patch)

    for name in pos:
        if name == "Y":
            continue
        arrow("Y", name, "#b7c3d1", 0.0)
    for src, dst in feature_edges:
        arrow(src, dst, "#d4762c", 0.12)

    for name, (x, y) in pos.items():
        if name == "Y":
            face = "#1f4e79"
        elif name in children:
            face = "#f4e1cc"
        else:
            face = "#d9e6f5"
        box = FancyBboxPatch(
            (x - 0.62, y - 0.22),
            1.24,
            0.44,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=face,
            edgecolor="#243140",
            linewidth=1.0,
        )
        ax.add_patch(box)
        color = "white" if name == "Y" else "#1c2833"
        ax.text(x, y, labels[name], ha="center", va="center", fontsize=9, color=color)

    ax.plot([], [], color="#b7c3d1", linewidth=2, label="类别 → 特征")
    ax.plot([], [], color="#d4762c", linewidth=2, label="特征 → 特征")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_scores(name, scores):
    print(
        f"{name}: acc={scores['accuracy']:.4f}  "
        f"P={scores['macro_precision']:.4f}  "
        f"R={scores['macro_recall']:.4f}  "
        f"F1={scores['macro_f1']:.4f}"
    )
    recalls = " ".join(f"{c}:{scores['per_class_recall'][c]:.3f}" for c in range(10))
    print(f"  recall {recalls}")


def main():
    parser = argparse.ArgumentParser(description="训练并评估两个手写数字分类器")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="MNIST idx 文件所在目录")
    args = parser.parse_args()

    ensure_mnist(args.data)
    x_train, y_train, x_test, y_test = load_mnist(args.data)
    print(f"train {x_train.shape} {y_train.shape}  test {x_test.shape} {y_test.shape}")

    nb = NaiveBayesClassifier(alpha=1.0).fit(x_train, y_train)
    nb_pred = nb.predict(x_test)
    nb_proba = nb.predict_proba(x_test)
    nb_cm = confusion_matrix(y_test, nb_pred)
    nb_scores = scores_from_cm(nb_cm)
    nb_pairs = top_confusions(nb_cm)
    nb_err = pick_error_indices(y_test, nb_pred, nb_proba, nb_pairs)

    bn = BayesNetClassifier(alpha=1.0).fit(x_train, y_train)
    bn_pred = bn.predict(x_test)
    bn_proba = bn.predict_proba(x_test)
    bn_cm = confusion_matrix(y_test, bn_pred)
    bn_scores = scores_from_cm(bn_cm)
    bn_pairs = top_confusions(bn_cm)
    bn_err = pick_error_indices(y_test, bn_pred, bn_proba, bn_pairs)

    print_scores("Naive Bayes", nb_scores)
    print("  confusions", nb_pairs)
    print_scores("Bayes Net  ", bn_scores)
    print("  confusions", bn_pairs)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(nb_cm, FIG_DIR / "nb_confusion.png", "朴素贝叶斯混淆矩阵")
    plot_confusion(bn_cm, FIG_DIR / "bn_confusion.png", "贝叶斯网络混淆矩阵")
    plot_errors(x_test, y_test, nb_pred, nb_err, FIG_DIR / "nb_errors.png", "朴素贝叶斯误分类样例")
    plot_errors(x_test, y_test, bn_pred, bn_err, FIG_DIR / "bn_errors.png", "贝叶斯网络误分类样例")
    plot_metrics(nb_scores, bn_scores, FIG_DIR / "metrics_compare.png")
    plot_top_pairs(nb_pairs, bn_pairs, FIG_DIR / "top_confusions.png")
    plot_structure(FIG_DIR / "bayes_net_structure.png")

    payload = {
        "test_size": int(len(y_test)),
        "naive_bayes": nb_scores,
        "bayes_net": bn_scores,
        "naive_bayes_confusions": nb_pairs,
        "bayes_net_confusions": bn_pairs,
        "naive_bayes_error_indices": nb_err,
        "bayes_net_error_indices": bn_err,
        "naive_bayes_error_detail": [
            {"index": i, "true": int(y_test[i]), "pred": int(nb_pred[i])} for i in nb_err
        ],
        "bayes_net_error_detail": [
            {"index": i, "true": int(y_test[i]), "pred": int(bn_pred[i])} for i in bn_err
        ],
        "features": list(FEATURE_NAMES),
        "feature_parents": [list(p) for p in FEATURE_PARENTS],
    }
    (ROOT / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"figures -> {FIG_DIR}")
    print(f"results -> {ROOT / 'results.json'}")


if __name__ == "__main__":
    main()
