"""伯努利朴素贝叶斯，用于 28×28 手写数字。

像素在模型内部二值化。概率用拉普拉斯平滑估计，预测用对数概率，避免 784 次连乘下溢。
"""

import numpy as np


class NaiveBayesClassifier:
    """Naive Bayes Protocol v1 —— 课程评测程序将按此接口加载并调用你的模型。"""

    def __init__(self, alpha: float = 1.0):
        """
        :param alpha: 平滑系数。默认为 1.0，对应 Laplace smoothing。
        """
        self.alpha = float(alpha)
        self.threshold = 128
        self.n_classes = 10
        self.n_features = 784
        self.class_count_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.feature_log_neg_prob_ = None

    def fit(self, X, y):
        """
        使用训练数据训练模型。

        :param X: shape 为 (n_samples, 784) 的图像特征矩阵。
        :param y: shape 为 (n_samples,) 的数字标签，取值为 0–9。
        :return: 返回 self。
        """
        X = _as_flat(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数不一致")
        if X.shape[1] != self.n_features:
            raise ValueError("每张图像应为 784 维")

        binary = X > self.threshold
        class_count = np.bincount(y, minlength=self.n_classes).astype(np.float64)
        on_count = np.zeros((self.n_classes, self.n_features), dtype=np.float64)
        for c in range(self.n_classes):
            mask = y == c
            if np.any(mask):
                on_count[c] = binary[mask].sum(axis=0)

        # 二值特征有 0/1 两种取值，拉普拉斯平滑的分母是 N_c + 2α
        denom = class_count[:, None] + 2.0 * self.alpha
        prob_on = (on_count + self.alpha) / denom
        prob_on = np.clip(prob_on, 1e-12, 1.0 - 1e-12)
        prior = (class_count + self.alpha) / (class_count.sum() + self.alpha * self.n_classes)

        self.class_count_ = class_count
        self.class_log_prior_ = np.log(prior)
        self.feature_log_prob_ = np.log(prob_on)
        self.feature_log_neg_prob_ = np.log(1.0 - prob_on)
        return self

    def _log_joint(self, X):
        if self.class_log_prior_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit")
        X = _as_flat(X)
        binary = (X > self.threshold).astype(np.float64)
        # log P(x|y) = Σ_j log P(x_j=0|y) + Σ_j x_j [log P(x_j=1|y) - log P(x_j=0|y)]
        base = self.feature_log_neg_prob_.sum(axis=1)
        delta = self.feature_log_prob_ - self.feature_log_neg_prob_
        return self.class_log_prior_[None, :] + base[None, :] + binary @ delta.T

    def predict(self, X):
        """
        预测输入图像的数字类别。

        :param X: shape 为 (n_samples, 784) 的图像特征矩阵。
        :return: shape 为 (n_samples,) 的预测标签数组。
        """
        return np.argmax(self._log_joint(X), axis=1).astype(np.int64)

    def predict_proba(self, X):
        """返回每个类别对应的归一化后验概率，形状 (n_samples, 10)。"""
        log_joint = self._log_joint(X)
        log_joint = log_joint - log_joint.max(axis=1, keepdims=True)
        prob = np.exp(log_joint)
        prob /= prob.sum(axis=1, keepdims=True)
        return prob


def _as_flat(X):
    X = np.asarray(X)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    elif X.ndim == 1:
        X = X.reshape(1, -1)
    return X
