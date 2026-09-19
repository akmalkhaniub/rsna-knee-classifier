"""End-to-end training loop test — skips cleanly where PyTorch is unavailable
(e.g. Python 3.14 with no wheel); runs on Kaggle / any torch-capable env."""
import numpy as np
import pytest

from rsnaknee import make_dataset


def test_synthetic_dataset_is_learnable_and_ordinal():
    x, y = make_dataset(n=400, seed=1)
    assert x.shape == (400, 6)
    assert set(np.unique(y)).issubset(set(range(5)))
    assert y.min() == 0 and y.max() == 4  # spans all 5 KL grades


def test_training_loop_learns_signal():
    torch = pytest.importorskip("torch")  # noqa: F841
    from rsnaknee import train_and_score
    r = train_and_score(epochs=80, seed=0)
    # The model must learn the synthetic signal: QWK well above chance (0).
    assert r.val_qwk > 0.5, f"val QWK too low: {r.val_qwk}"
    assert r.n_train > 0 and r.n_val > 0
