"""Competition metrics for RSNA knee grading.

- ``quadratic_weighted_kappa`` for ordinal KL-grade agreement.
- ``weighted_log_loss`` for the multi-label abnormality probabilities.

These let the pipeline report a real score on any labeled split (validation / OOF),
which is the number that decides the competition.
"""
from __future__ import annotations

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int | None = None) -> float:
    """Cohen's kappa with quadratic weights, for ordinal labels (e.g. KL grades 0..4)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return 0.0
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    # Observed confusion matrix.
    O = np.zeros((n_classes, n_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    # Quadratic weight matrix.
    idx = np.arange(n_classes)
    W = (idx[:, None] - idx[None, :]) ** 2 / (n_classes - 1) ** 2
    # Expected matrix from marginals.
    act = O.sum(axis=1)
    pred = O.sum(axis=0)
    E = np.outer(act, pred) / O.sum()
    denom = (W * E).sum()
    if denom == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / denom)


def weighted_log_loss(y_true, y_prob, weights=None, eps: float = 1e-7) -> float:
    """Mean binary cross-entropy across labels, optionally per-label weighted.

    y_true, y_prob: arrays of shape (n_samples, n_labels).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    bce = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))  # (n, L)
    per_label = bce.mean(axis=0)  # (L,)
    if weights is None:
        return float(per_label.mean())
    w = np.asarray(weights, dtype=float)
    return float((per_label * w).sum() / w.sum())
