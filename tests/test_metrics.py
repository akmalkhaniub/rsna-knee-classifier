"""RSNA competition-metric tests (QWK + weighted log-loss)."""
import numpy as np
from rsnaknee import quadratic_weighted_kappa, weighted_log_loss


def test_qwk_perfect_agreement():
    y = [0, 1, 2, 3, 4, 2, 1]
    assert abs(quadratic_weighted_kappa(y, y, n_classes=5) - 1.0) < 1e-9


def test_qwk_penalizes_far_errors_more_than_near():
    y = [0, 0, 4, 4]
    near = [0, 1, 3, 4]   # off by 1
    far = [4, 4, 0, 0]    # maximally wrong
    k_near = quadratic_weighted_kappa(y, near, n_classes=5)
    k_far = quadratic_weighted_kappa(y, far, n_classes=5)
    assert k_near > k_far
    assert k_far <= 0.0  # anti-correlated predictions


def test_qwk_empty_safe():
    assert quadratic_weighted_kappa([], [], n_classes=5) == 0.0


def test_weighted_log_loss_lower_is_better():
    y = np.array([[1, 0, 1], [0, 1, 0]])
    good = np.array([[0.9, 0.1, 0.8], [0.2, 0.85, 0.1]])
    bad = np.array([[0.1, 0.9, 0.2], [0.8, 0.15, 0.9]])
    assert weighted_log_loss(y, good) < weighted_log_loss(y, bad)


def test_weighted_log_loss_weights_applied():
    y = np.array([[1, 0]])
    prob = np.array([[0.6, 0.6]])
    # Weighting label 0 heavily changes the aggregate vs. equal weights.
    equal = weighted_log_loss(y, prob)
    weighted = weighted_log_loss(y, prob, weights=[5, 1])
    assert weighted != equal
