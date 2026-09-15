# RSNA Knee Abnormality Detection

- **Official Challenge URL:** [https://www.kaggle.com/competitions](https://www.kaggle.com/competitions)
- **Organizer:** Radiological Society of North America (RSNA) & Kaggle
- **Host Platform:** Kaggle Competitions
- **Total Prize Pool:** $77,000 USD
- **Format:** Code Competition (Notebook execution on Kaggle GPU)
- **Primary Themes:** Medical Imaging, 3D Volumetric Computer Vision, Multi-Sequence MRI, Deep Learning

---

## 1. Challenge Overview & Problem Statement
Knee injuries (such as ACL tears, meniscus tears, and cartilage abnormalities) are among the most common musculoskeletal complaints evaluated via Magnetic Resonance Imaging (MRI). 

Radiologists evaluate multiple views (sagittal, coronal, axial) and sequences (T1, T2, proton density fat-suppressed) to detect structural abnormalities. This competition requires building high-sensitivity, multi-modal 3D deep learning models that ingest multi-sequence volumetric DICOM scans and predict the presence and severity of specific knee abnormalities.

### Evaluation Metric
Submissions are evaluated on **Weighted Multi-Label Log Loss / Area Under the ROC Curve (AUC)** across target abnormality classes.

---

## 2. Selected Architectural Strategy: 2.5D/3D Multi-View Vision Transformer Ensemble
1. **Volumetric Preprocessing:** Normalization of voxel intensity, slice interpolation to standardized physical spacing, and automatic knee joint cropping.
2. **Backbone Architectures:**
   - Pre-trained 3D CNNs (e.g. 3D ResNet, Med3D).
   - 2.5D Slice Encoders (Swin Transformer / CoAtNet) paired with temporal/spatial sequence aggregators (Bidirectional GRU / Multi-Head Self-Attention across slice depth).
3. **Cross-View Multi-Modal Fusion:** Late fusion network combining sagittal, coronal, and axial sequence embeddings into a single multi-label classification head.

---

## 3. Directory Structure
```
kaggle-rsna-knee/
├── README.md               # Challenge overview, DICOM details, metrics (this file)
├── SPECIFICATION.md        # Preprocessing pipeline, model architectures, loss formulations
├── ROADMAP.md              # Cross-validation, training, and submission milestones
├── src/
│   ├── dataset.py          # DICOM loader, 3D augmentation, and volumetric sampling
│   ├── models.py           # 2.5D Swin & 3D CNN backbones with multi-view fusion
│   ├── losses.py           # Class-weighted focal loss and multi-label BCE
│   └── train.py            # Distributed training loop with mixed precision (AMP)
└── notebooks/              # Inference notebook for Kaggle code submission
```
