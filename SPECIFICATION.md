# Technical Specification: 3D Multi-View MRI Knee Classifier
**Project Name:** 3D Multi-View MRI Knee Classifier (RSNA Knee Abnormality Detection)  
**Status:** Rebuilt in Python — produces an RSNA submission (updated 2026-09-18)  
**Version:** 1.0.0  

> **Implementation status (2026-09-18):** Rebuilt from the Node prototype (now under `legacy-js/`) into a Python package (`rsnaknee/`): real NumPy volumetric preprocessing, a heuristic multi-view classifier baseline, an optional PyTorch 2.5D multi-view fusion model scaffold, and an RSNA `submission.csv` builder + schema validator. 6 pytest cases pass. Not built: trained model weights, DICOM cross-validation, and calibration; the default runs weight-free.

---

## 1. Pipeline Architecture
The system accepts raw volumetric MRI DICOM series from three standard planes (sagittal, coronal, axial). Each sequence is processed through a spatial feature encoder, aggregated across slice depth, and fused into a joint representation for multi-label prediction.

```mermaid
graph TD
    A[Raw DICOM Series: Sagittal, Coronal, Axial] --> B[Voxel Resampling & HU / Intensity Normalization]
    B --> C[3D Volumetric Augmentation (Flip, Rotation, Elastic Deformation)]
    C --> D1[Sagittal 2.5D Swin Transformer Backbone]
    C --> D2[Coronal 2.5D Swin Transformer Backbone]
    C --> D3[Axial 2.5D Swin Transformer Backbone]
    D1 --> E[Sequence Depth Aggregator: Self-Attention]
    D2 --> E
    D3 --> E
    E --> F[Cross-View Late Fusion Layer]
    F --> G[Multi-Label Classification Head: ACL, Meniscus, Cartilage]
    G --> H[Sigmoid Probability Output & Calibrated Thresholds]
```

---

## 2. Technical Specifications

### 2.1 Volumetric Preprocessing Pipeline
- **Intensity Normalization:** Percentile clipping (1st to 99th percentile) followed by z-score standardization.
- **Slice Sampling:** Uniform interpolation to 32 slices per sequence volume `(C=1, D=32, H=256, W=256)`.
- **Data Augmentation:** Random 3D affine transformations, subtle gamma adjustments, and Gaussian noise via `albumentations` / `monai`.

### 2.2 Model Architecture
- **Per-Slice Encoder:** Pretrained 2D Swin-Transformer-Base extracting a 1024-dimensional feature vector per slice.
- **Temporal/Depth Pooling:** Multi-head self-attention layer aggregating `(32, 1024)` vectors into a single volume representation `(1024,)`.
- **Fusion:** Concatenation of the 3 sequence embeddings `(3072,)` followed by Dropout (0.3), LayerNorm, and a Linear projection layer to `num_classes`.

### 2.3 Loss Function
- Asymmetric Loss (ASL) / Multi-Label Focal Loss to handle severe class imbalance:
  $$\mathcal{L} = -\sum_{k} \left[ y_k (1 - p_k)^{\gamma_+} \log(p_k) + (1 - y_k) (p_k - m)_+^{\gamma_-} \log(1 - p_k + m) \right]$$

---

## 3. Performance & Resource Constraints
- **GPU Inference Budget:** Kaggle enforces a strict 9-hour limit for all test cases. Inference per patient study must execute in `< 1.2 seconds` on an NVIDIA T4 / P100.
- **Batch Size & Precision:** PyTorch Automatic Mixed Precision (FP16 / BF16) with gradient accumulation.

---

## 4. Acceptance Criteria
1. Full 5-fold cross-validation scheme stratified by patient ID to prevent data leakage.
2. Training loss steadily converges with validation AUC consistently exceeding 0.91 across classes.
3. Fully standalone Kaggle submission notebook that runs offline without internet access.
