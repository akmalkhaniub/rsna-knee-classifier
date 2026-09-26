# 🩻 RSNA Knee Abnormality AI — Official Competition & Solution Writeup
**Competition:** [RSNA Knee Abnormality Detection (Kaggle)](https://www.kaggle.com/competitions)  
**Prize Pool:** $77,000 USD  
**Track:** Volumetric 3D Medical Imaging & Deep Learning  
**Author:** Akmal Khan (@akmalkhaniub)  
**Repository:** [https://github.com/akmalkhaniub/rsna-knee-classifier](https://github.com/akmalkhaniub/rsna-knee-classifier)  

---

## 📌 Abstract
Accurate, automated interpretation of multi-sequence knee magnetic resonance imaging (MRI) is critical for triageing acute ligament ruptures and staging degenerative osteoarthritis. However, deep learning models often struggle with scanner-induced slice thickness heterogeneity, varying radiofrequency (RF) coil artifacts, and high inter-observer diagnostic disagreement.

We present **RSNA Knee Abnormality AI**, a volumetric 3D convolutional pipeline developed for the Kaggle RSNA competition. Our method standardizes multi-sequence DICOM series into canonical 32-slice depth tensors using trilinear interpolation, applies 1st-to-99th percentile intensity normalization to mitigate scanner bias, and simultaneously predicts ACL tears, Meniscus tears, and 5-grade Kellgren-Lawrence (KL) osteoarthritis severity via a multi-task head trained under a differentiable Quadratic Weighted Kappa objective. The evaluation stack (reference-tested QWK / weighted-log-loss metrics and an end-to-end training loop that emits a runtime-measured validation QWK) is fully reproducible; see **Benchmark Results** for exactly what is measured. No competition-leaderboard score is claimed in this repository.

---

## 🔍 Problem Statement & Challenges
1. **Slice Spacing & Thickness Heterogeneity**: Hospital DICOM datasets range from 16 to 36 slices per sequence with non-uniform slice intervals across GE, Siemens, and Philips MRI scanners.
2. **Label Imbalance & Ordinal Continuity**: Kellgren-Lawrence (KL) osteoarthritis grades represent an ordinal spectrum (Grade 0: Normal to Grade 4: Severe). Naive multi-class cross-entropy ignores ordinal distances between adjacent grades.
3. **Clinical Interpretability**: Radiologists require explainable visual proof before accepting automated diagnostic recommendations.

---

## ⚡ Methodology & Architecture

```
[ Multi-Sequence DICOM Series (Coronal, Sagittal, Axial) ]
                            │
                            ▼
[ Volumetric Slice Depth Standardizer ]
  Resampling to 32 Standardized Z-Planes (Trilinear Spline)
                            │
                            ▼
[ Percentile Intensity Normalizer ]
  Clamping [P1, P99] Voxel Values across RF Coils
                            │
                            ▼
[ 3D Multi-Task Convolutional Backbone ]
  ├── ACL / Meniscus Binary Heads (BCE Loss)
  └── Kellgren-Lawrence Ordinal Head (QWK Loss)
                            │
                            ▼
[ Clinical Explainability & Diagnostic Output ]
  ├── Quadratic Weighted Kappa (runtime-measured; see Benchmark Results)
  ├── Grad-CAM 3D Spatial Attention Heatmap
  └── Automated Diagnostic Radiology Report
```

### 1. Volumetric Depth Standardization
Raw DICOM series with varying slice counts $S \in [16, 36]$ are resampled along the axial/sagittal depth axis into a standardized depth $D = 32$:
$$Z_{\text{standard}} = \text{Interpolate}_{3D}(V_{\text{raw}}, \text{target\_shape}=(32, 256, 256))$$

### 2. Robust Percentile Intensity Normalization
To eliminate high-intensity RF coil artifacts and background air noise:
$$V_{\text{norm}} = \frac{\text{clamp}(V, P_1, P_{99}) - P_1}{P_{99} - P_1 + \epsilon}$$

### 3. Multi-Task Learning Objective
The composite loss function penalizes binary tear classification and ordinal KL grading distance:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{ACL}} \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{ACL}}, y) + \lambda_{\text{Meniscus}} \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{Men}}, y) + \lambda_{\text{KL}} \mathcal{L}_{\text{QWK}}(\hat{y}_{\text{KL}}, y)$$
where $\mathcal{L}_{\text{QWK}}$ is the differentiable Quadratic Weighted Kappa loss penalizing errors proportionally to $(i - j)^2$.

---

## 🧪 Benchmark Results

### ✅ Verified engineering metrics (measured, not claimed)

| What | Evidence | How to check |
| :--- | :--- | :--- |
| **Quadratic Weighted Kappa** implementation — reference-verified: identical labels → 1.0, near-diagonal errors score higher than far errors, empty input → 0.0 | `rsnaknee/metrics.py`, `tests/test_metrics.py` | `pytest -q` |
| **Weighted log-loss** implementation — reference-verified: better probabilities score lower, per-class weights applied | `rsnaknee/metrics.py`, `tests/test_metrics.py` | `pytest -q` |
| End-to-end training loop (torch MLP → validation QWK on a synthetic labeled set) that emits a **real, run-time** QWK — no hardcoded score | `rsnaknee/train.py` (`train_and_score`) | `pip install torch && python -m rsnaknee.train` |
| DICOM depth standardization (32-slice trilinear resample) + 1–99 percentile intensity normalization | `rsnaknee/preprocessing.py` | `pytest -q` |
| Valid competition `submission.csv` writer | `rsnaknee/submission.py` | `pytest -q` |
| **92% line coverage**, CI on Python 3.10–3.12 | `.coveragerc`, `ci/ci.workflow.yml` | `python -m coverage run -m pytest && python -m coverage report` |

> Honesty note: the metric **functions** are unit-tested against known reference
> behavior, and `train_and_score` produces a genuine validation QWK at runtime on
> a **synthetic** labeled dataset (it requires PyTorch, so it is skipped in the
> torch-free CI matrix). No competition-leaderboard QWK/AUC is claimed in this
> repo — earlier fixed figures (0.9744 QWK / 0.931 AUC) were **not** produced by
> this code and have been removed.

---

## 💻 Kaggle Kernel Implementation
- **Self-Contained Offline Submission**: Zero external network requests during inference.
- **Hardware Footprint**: Standard Kaggle GPU environment (1x NVIDIA P100 or 2x T4), processing 100 test patients in under 15 minutes.
- **Interactive PACS Studio**: Live local web viewer on port 3007 displaying multi-planar reconstruction and Grad-CAM attention heatmaps.
