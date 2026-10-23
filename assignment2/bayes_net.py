"""手工小型贝叶斯网络，用于 28×28 手写数字。

类别 Y 是根节点。12 个离散特征里有 7 条特征之间的有向边。
每个特征节点最多再有 1 个特征父节点，因此父节点总数不超过 2。
参数是带拉普拉斯平滑的条件概率表。预测时特征全部被观测，直接比较对数联合概率。
"""

import numpy as np
from scipy import ndimage

N_CLASSES = 10
N_FEATURES = 12
CARD = 3  # 每个特征都离散成 3 档

# 特征顺序固定，extract_features 的列与这里一致。
FEATURE_NAMES = (
    "top",
    "mid",
    "bot",
    "left",
    "right",
    "aspect",
    "h_sym",
    "v_sym",
    "holes",
    "hole_pos",
    "cross",
    "profile",
)

# 每个特征除了类别 Y 之外的父节点（特征下标）。Y 是所有特征的父节点，不写在这里。
FEATURE_PARENTS = (
    (),     # top
    (0,),   # mid | top          竖向笔画从上向下延续
    (1,),   # bot | mid
    (),     # left
    (3,),   # right | left       左右墨迹受同一支笔的轻重影响
    (),     # aspect
    (3,),   # h_sym | left       左侧墨迹偏多时左右更不对称
    (0,),   # v_sym | top        上方墨迹偏多时上下更不对称
    (),     # holes
    (8,),   # hole_pos | holes   没有孔就没有孔的位置
    (1,),   # cross | mid        中部有没有墨，决定中线能穿越几次
    (),     # profile
)

# 连续特征的分档边界（低 < lo，中，高 >= hi）。
# 先按训练集分布取整，写死之后训练和测试用同一套边界。
BIN_EDGES = {
    "top": (0.12, 0.20),
    "mid": (0.16, 0.24),
    "bot": (0.10, 0.18),
    "left": (0.12, 0.22),
    "right": (0.14, 0.24),
    "aspect": (0.62, 0.85),
    "h_sym": (0.83, 0.87),
    "v_sym": (0.82, 0.86),
    "profile": (-1.2, 1.5),
}

# 孔洞中心在字高中的相对位置。小于该值视为靠上，否则视为靠下或居中。
HOLE_UPPER = 0.46

INK_THRESHOLD = 128
MIN_HOLE_PIXELS = 4
_STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)


def extract_features(X):
    """
    从原始 784 维图像中提取离散特征。

    :param X: shape 为 (n_samples, 784) 的原始图像特征矩阵。
    :return: shape 为 (n_samples, k) 的离散特征矩阵。
    """
    raw = raw_features(X)
    feats = np.empty((raw["top"].shape[0], N_FEATURES), dtype=np.int64)
    for index, name in enumerate(FEATURE_NAMES):
        if name in BIN_EDGES:
            lo, hi = BIN_EDGES[name]
            feats[:, index] = _bin3(raw[name], lo, hi)
        elif name == "holes":
            feats[:, index] = np.clip(raw[name], 0, 2)
        elif name == "hole_pos":
            feats[:, index] = _bin_hole_pos(raw["holes"], raw["hole_rel"])
        elif name == "cross":
            feats[:, index] = _bin_cross(raw[name])
        else:
            raise KeyError(name)
    return feats


def raw_features(X):
    """返回离散化之前的特征，便于检查分布。键为特征名。"""
    images = _as_images(X)
    binary = images > INK_THRESHOLD
    # 四周空白不计入区域密度，窗口盖住的是居中的字身。
    top = binary[:, 4:12, :].mean(axis=(1, 2))
    mid = binary[:, 12:18, :].mean(axis=(1, 2))
    bot = binary[:, 18:26, :].mean(axis=(1, 2))
    left = binary[:, :, 6:12].mean(axis=(1, 2))
    right = binary[:, :, 16:22].mean(axis=(1, 2))
    top_width = binary[:, 6:12, :].sum(axis=2).mean(axis=1)
    bot_width = binary[:, 18:24, :].sum(axis=2).mean(axis=1)

    left_half = binary[:, :, :14].astype(np.float32)
    right_half = binary[:, :, 14:][:, :, ::-1].astype(np.float32)
    h_sym = 1.0 - np.abs(left_half - right_half).mean(axis=(1, 2))

    top_half = binary[:, :14, :].astype(np.float32)
    bot_half = binary[:, 14:, :][:, ::-1, :].astype(np.float32)
    v_sym = 1.0 - np.abs(top_half - bot_half).mean(axis=(1, 2))

    holes, hole_row = _holes_and_centroids(binary)
    bbox_top, bbox_height = _bbox_top_height(binary)
    hole_rel = np.full(holes.shape[0], -1.0, dtype=np.float64)
    has_hole = holes > 0
    hole_rel[has_hole] = (hole_row[has_hole] - bbox_top[has_hole]) / bbox_height[has_hole]

    return {
        "top": top,
        "mid": mid,
        "bot": bot,
        "left": left,
        "right": right,
        "aspect": _aspect_ratio(binary),
        "h_sym": h_sym.astype(np.float64),
        "v_sym": v_sym.astype(np.float64),
        "holes": holes,
        "hole_rel": hole_rel,
        "cross": _middle_crossings(binary),
        "profile": (top_width - bot_width).astype(np.float64),
    }


class BayesNetClassifier:
    """Small Bayesian Network Protocol v1 —— 课程评测程序将按此接口加载并调用你的模型。"""

    def __init__(self, alpha: float = 1.0):
        """
        :param alpha: 平滑系数。默认为 1.0，对应 Laplace smoothing。
        """
        self.alpha = float(alpha)
        self.class_log_prior_ = None
        self.log_cpts_ = None

    def fit(self, X, y):
        """
        使用训练数据训练贝叶斯网络。

        :param X: shape 为 (n_samples, 784) 的原始图像特征矩阵。
        :param y: shape 为 (n_samples,) 的数字标签，取值为 0–9。
        :return: 返回 self。
        """
        y = np.asarray(y, dtype=np.int64).ravel()
        feats = extract_features(X)
        if feats.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数不一致")

        class_count = np.bincount(y, minlength=N_CLASSES).astype(np.float64)
        prior = (class_count + self.alpha) / (class_count.sum() + self.alpha * N_CLASSES)
        self.class_log_prior_ = np.log(prior)

        log_cpts = []
        for index, parents in enumerate(FEATURE_PARENTS):
            values = feats[:, index]
            if not parents:
                counts = np.zeros((N_CLASSES, CARD), dtype=np.float64)
                np.add.at(counts, (y, values), 1.0)
                smoothed = counts + self.alpha
                probs = smoothed / smoothed.sum(axis=1, keepdims=True)
            else:
                parent_values = feats[:, parents[0]]
                counts = np.zeros((N_CLASSES, CARD, CARD), dtype=np.float64)
                np.add.at(counts, (y, parent_values, values), 1.0)
                smoothed = counts + self.alpha
                probs = smoothed / smoothed.sum(axis=2, keepdims=True)
            log_cpts.append(np.log(probs))
        self.log_cpts_ = log_cpts
        return self

    def _log_joint(self, X):
        if self.class_log_prior_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        feats = extract_features(X)
        n = feats.shape[0]
        log_joint = np.tile(self.class_log_prior_, (n, 1))
        for index, parents in enumerate(FEATURE_PARENTS):
            values = feats[:, index]
            log_cpt = self.log_cpts_[index]
            if not parents:
                # log_cpt[:, values] 的形状是 (类别数, 样本数)
                log_joint += log_cpt[:, values].T
            else:
                parent_values = feats[:, parents[0]]
                log_joint += log_cpt[:, parent_values, values].T
        return log_joint

    def predict(self, X):
        """
        预测输入图像的数字类别。

        :param X: shape 为 (n_samples, 784) 的原始图像特征矩阵。
        :return: shape 为 (n_samples,) 的预测标签数组。
        """
        return np.argmax(self._log_joint(X), axis=1).astype(np.int64)

    def predict_proba(self, X):
        """返回归一化后验概率，形状 (n_samples, 10)。"""
        log_joint = self._log_joint(X)
        log_joint = log_joint - log_joint.max(axis=1, keepdims=True)
        prob = np.exp(log_joint)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob


def _as_images(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim == 2:
        X = X.reshape(X.shape[0], 28, 28)
    if X.shape[1:] != (28, 28):
        raise ValueError("图像应为 28×28 或展平后的 784 维")
    return X


def _bin3(values, lo, hi):
    out = np.ones(values.shape[0], dtype=np.int64)
    out[values < lo] = 0
    out[values >= hi] = 2
    return out


def _bin_hole_pos(holes, hole_rel):
    """无孔 → 0，孔心在字高上半 → 1，否则 → 2。"""
    out = np.zeros(holes.shape[0], dtype=np.int64)
    has = holes > 0
    out[has & (hole_rel < HOLE_UPPER)] = 1
    out[has & (hole_rel >= HOLE_UPPER)] = 2
    return out


def _bin_cross(counts):
    """中线穿越 0–1 次 → 0，2 次 → 1，3 次及以上 → 2。"""
    out = np.full(counts.shape, 2, dtype=np.int64)
    out[counts <= 1] = 0
    out[counts == 2] = 1
    return out


def _aspect_ratio(binary):
    rows = binary.any(axis=2)
    cols = binary.any(axis=1)
    top = np.argmax(rows, axis=1)
    bot = binary.shape[1] - 1 - np.argmax(rows[:, ::-1], axis=1)
    left = np.argmax(cols, axis=1)
    right = binary.shape[2] - 1 - np.argmax(cols[:, ::-1], axis=1)
    height = np.maximum(bot - top + 1, 1)
    width = np.maximum(right - left + 1, 1)
    aspect = width / height
    aspect[~rows.any(axis=1)] = 1.0
    return aspect.astype(np.float64)


def _middle_crossings(binary):
    row = binary[:, 14, :].astype(np.int8)
    previous = np.zeros((binary.shape[0], 1), dtype=np.int8)
    padded = np.concatenate([previous, row], axis=1)
    return (np.diff(padded, axis=1) == 1).sum(axis=1).astype(np.int64)


def _bbox_top_height(binary):
    rows = binary.any(axis=2)
    top = np.argmax(rows, axis=1).astype(np.float64)
    bot = binary.shape[1] - 1 - np.argmax(rows[:, ::-1], axis=1)
    height = np.maximum(bot - top + 1, 1).astype(np.float64)
    empty = ~rows.any(axis=1)
    top[empty] = 0
    height[empty] = float(binary.shape[1])
    return top, height


def _holes_and_centroids(binary, min_hole=MIN_HOLE_PIXELS):
    """数封闭孔洞，并给出含有孔洞像素的那些行的平均行号。

    多张图拼到一张带空白间隔的大图上，背景用 4 邻域一次标记完。
    和间隔相连的背景是字外面，其余小块才是孔。4 邻域配合字的斜向连接，避免孔被斜缝漏掉。
    """
    n, height, width = binary.shape
    holes = np.empty(n, dtype=np.int64)
    centroid = np.full(n, -1.0, dtype=np.float64)
    chunk = 1024
    cols = 32
    for start in range(0, n, chunk):
        batch = np.asarray(binary[start:start + chunk], dtype=bool)
        m = batch.shape[0]
        rows = (m + cols - 1) // cols
        tile = np.zeros((rows * (height + 1), cols * (width + 1)), dtype=bool)
        origins = []
        for i in range(m):
            r, c = divmod(i, cols)
            rs, cs = r * (height + 1), c * (width + 1)
            tile[rs:rs + height, cs:cs + width] = batch[i]
            origins.append((rs, cs))

        bg_labeled, _ = ndimage.label(~tile, structure=_STRUCT4)
        gap_rows = np.zeros(tile.shape[0], dtype=bool)
        gap_cols = np.zeros(tile.shape[1], dtype=bool)
        for r in range(rows):
            gap_rows[r * (height + 1) + height] = True
        for c in range(cols):
            gap_cols[c * (width + 1) + width] = True
        outside_labels = np.unique(np.concatenate([
            bg_labeled[gap_rows].ravel(),
            bg_labeled[:, gap_cols].ravel(),
        ]))
        outside = np.zeros(int(bg_labeled.max()) + 1, dtype=bool)
        outside[outside_labels] = True
        outside[0] = True
        for i, (rs, cs) in enumerate(origins):
            cell = bg_labeled[rs:rs + height, cs:cs + width]
            labels = np.unique(cell)
            keep = labels[~outside[labels]]
            if keep.size == 0:
                holes[start + i] = 0
                continue
            sizes = np.bincount(cell.ravel())
            keep = keep[sizes[keep] >= min_hole]
            holes[start + i] = int(keep.size)
            if keep.size == 0:
                continue
            mask = np.isin(cell, keep)
            centroid[start + i] = float(np.flatnonzero(mask.any(axis=1)).mean())
    return holes, centroid
