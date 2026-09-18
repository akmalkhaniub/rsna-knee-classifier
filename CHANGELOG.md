# Changelog

## [Unreleased]

### Rebuilt in Python for Kaggle (2026-09-18)
Rebuilt from the Node prototype (preserved under `legacy-js/`) into a Python package that
produces an RSNA `submission.csv`.

### Added
- `rsnaknee/preprocessing.py` — NumPy depth standardization + percentile/z-score
  normalization; optional pydicom series loader.
- `rsnaknee/model.py` — `HeuristicMultiViewClassifier` baseline (no training) and an
  optional PyTorch 2.5D multi-view fusion model scaffold (`build_torch_model`).
- `rsnaknee/submission.py` — build + validate the RSNA submission CSV (stdlib only).
- `notebooks/kaggle_run.py` — Kaggle entry point (wire `load_studies` to the data layout).
- `tests/test_rsnaknee.py` — 6 pytest cases (resampling, normalization, probability
  range/monotonicity, submission round-trip + schema validation).
- `pyproject.toml`, `requirements*.txt`, CI on Python 3.10–3.12.

### Notes
- Default classifier is a heuristic baseline so the pipeline runs with no weights/GPU;
  training the torch model on the DICOM set is the path to a competitive score.
