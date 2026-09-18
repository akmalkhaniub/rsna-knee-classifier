"""RSNA knee tests: preprocessing, classifier probabilities, and submission schema."""
import numpy as np

from rsnaknee import (
    HeuristicMultiViewClassifier,
    LABELS,
    normalize_intensity,
    predict_studies,
    preprocess_volume,
    read_submission,
    standardize_depth,
    validate_submission_rows,
    write_submission,
)


def test_standardize_depth_up_and_down():
    vol = np.random.rand(10, 8, 8).astype(np.float32)
    assert standardize_depth(vol, 32).shape == (32, 8, 8)
    assert standardize_depth(vol, 5).shape == (5, 8, 8)
    assert standardize_depth(np.random.rand(32, 4, 4), 32).shape == (32, 4, 4)


def test_normalize_intensity_zscore():
    vol = np.random.rand(4, 16, 16).astype(np.float32) * 1000
    out = normalize_intensity(vol)
    assert abs(float(out.mean())) < 0.2   # roughly zero-centered
    assert 0.5 < float(out.std()) < 2.0


def test_preprocess_pipeline_shape():
    assert preprocess_volume(np.random.rand(7, 12, 12), target_depth=16).shape == (16, 12, 12)


def test_classifier_probabilities_in_range():
    clf = HeuristicMultiViewClassifier()
    pred = clf.predict({"study_id": "s1", "sagittal": np.ones((4, 4)), "coronal": np.ones((4, 4)), "axial": np.ones((4, 4))})
    assert set(pred.probabilities) == set(LABELS)
    assert all(0.0 <= v <= 1.0 for v in pred.probabilities.values())
    assert isinstance(pred.impression["acl_tear"], bool)


def test_classifier_monotonic_signal():
    clf = HeuristicMultiViewClassifier()
    low = clf.predict({"study_id": "lo", "sagittal": np.zeros((4, 4))}).probabilities["acl_tear"]
    high = clf.predict({"study_id": "hi", "sagittal": np.ones((4, 4)) * 2}).probabilities["acl_tear"]
    assert high > low  # stronger sagittal signal -> higher ACL probability


def test_submission_roundtrip_and_validation(tmp_path):
    studies = [
        {"study_id": "a", "sagittal": np.ones((4, 4)), "coronal": np.ones((4, 4)), "axial": np.ones((4, 4))},
        {"study_id": "b", "sagittal": np.zeros((4, 4)), "coronal": np.zeros((4, 4)), "axial": np.zeros((4, 4))},
    ]
    preds = predict_studies(studies)
    path = write_submission(preds, tmp_path / "submission.csv")
    rows = read_submission(path)
    assert len(rows) == 2
    validate_submission_rows(rows)  # raises on any malformed row
