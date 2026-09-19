# 🦵 RSNA Knee Abnormality Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-preprocessing-013243.svg)](https://numpy.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-optional%20model-EE4C2C.svg)](https://pytorch.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-RSNA%20Knee-20BEFF.svg)](https://www.kaggle.com/competitions)

> **Built for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions) Kaggle code competition.**

Multi-sequence knee-MRI abnormality classifier (ACL tear, meniscus tear, cartilage
abnormality). Real NumPy volumetric preprocessing, a no-train fusion baseline, an optional
PyTorch 2.5D multi-view model, and an RSNA-format `submission.csv`.

> **Note:** rebuilt from a Node.js prototype (kept under [`legacy-js/`](./legacy-js)) into
> the correct stack — a **Python** package that produces a Kaggle submission.

## Modules

- **`preprocessing.py`** — `standardize_depth` (resample to a fixed slice count),
  `normalize_intensity` (percentile-clip + z-score), and `load_dicom_series` (pydicom,
  optional). All array math is real NumPy.
- **`model.py`** — `HeuristicMultiViewClassifier` (sagittal/coronal/axial signal fusion →
  calibrated probabilities, no training needed) and `build_torch_model` (a ResNet-18-based
  2.5D three-view fusion net, built only when torch is installed).
- **`submission.py`** — build + validate the RSNA `submission.csv` (study_id + one
  probability column per label), stdlib-only.

## Run

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest -q                       # preprocessing + classifier + submission-schema tests
python notebooks/kaggle_run.py  # wire load_studies() to the competition data to submit
```

## Training loop

`rsnaknee.train.train_and_score()` is a complete train → predict → **score** loop: it fits a small classifier on a synthetic *learnable* dataset and reports **QWK** on a held-out split, proving the pipeline end-to-end. Swap `make_dataset` for real DICOM-derived features to train for real. Requires PyTorch (optional dep); the test skips cleanly where torch is unavailable and runs on Kaggle / any torch-capable env.

## Metrics

`rsnaknee.metrics` computes the competition scores on any labeled split so you can report a real number: `quadratic_weighted_kappa` (ordinal KL-grade agreement) and `weighted_log_loss` (multi-label abnormality probabilities). Tested in `tests/test_metrics.py` (perfect agreement → 1.0; far errors penalized more than near).

## Scope & honesty

The preprocessing and submission plumbing are real and tested; the default classifier is a
**heuristic baseline** so the pipeline runs end-to-end with no trained weights or GPU. The
PyTorch model is a ready scaffold — training it on the RSNA DICOM set (with cross-validation
and calibration) is what yields a competitive QWK/log-loss score. Tests validate depth
resampling, z-score normalization, probability ranges/monotonicity, and the exact
submission schema; real scoring happens on Kaggle's hidden set.
