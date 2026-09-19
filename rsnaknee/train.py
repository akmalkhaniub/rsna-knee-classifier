"""End-to-end training loop (proof that the train → predict → score path works).

Trains a small classifier to predict the ordinal KL grade from per-sequence features,
then scores it with the real competition metric (quadratic weighted kappa) on a
held-out split. Uses a synthetic, learnable dataset so the loop runs fast with no data;
swap `make_dataset` for real DICOM-derived features to train for real.

Requires PyTorch (optional dependency). Import-safe without it: `train_and_score`
raises a clear error, and the test skips.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import quadratic_weighted_kappa

N_CLASSES = 5  # KL grades 0..4


def make_dataset(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (features -> KL grade) where the grade is a learnable function of x."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, 6)).astype(np.float32)
    # A monotonic score → binned into 5 ordinal grades, plus a little label noise.
    score = x[:, 0] * 1.5 + x[:, 1] - x[:, 2] * 0.8 + rng.normal(0, 0.3, n)
    edges = np.quantile(score, [0.2, 0.4, 0.6, 0.8])
    y = np.digitize(score, edges).astype(np.int64)  # 0..4
    return x, y


@dataclass
class TrainResult:
    epochs: int
    final_loss: float
    val_qwk: float
    n_train: int
    n_val: int


def train_and_score(epochs: int = 60, seed: int = 0) -> TrainResult:  # pragma: no cover - requires PyTorch (skipped where unavailable)
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("PyTorch is required for train_and_score (pip install torch).") from exc

    torch.manual_seed(seed)
    x, y = make_dataset(seed=seed)
    split = int(0.8 * len(x))
    xt, yt = torch.tensor(x[:split]), torch.tensor(y[:split])
    xv, yv = torch.tensor(x[split:]), torch.tensor(y[split:])

    model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, N_CLASSES))
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    final_loss = 0.0
    for _ in range(epochs):
        opt.zero_grad()
        out = model(xt)
        loss = loss_fn(out, yt)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

    with torch.no_grad():
        preds = model(xv).argmax(dim=1).numpy()
    val_qwk = quadratic_weighted_kappa(yv.numpy(), preds, n_classes=N_CLASSES)
    return TrainResult(epochs=epochs, final_loss=round(final_loss, 4), val_qwk=round(val_qwk, 4),
                       n_train=split, n_val=len(x) - split)


if __name__ == "__main__":  # pragma: no cover
    print(train_and_score())
