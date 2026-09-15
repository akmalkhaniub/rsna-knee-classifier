# Roadmap & Milestones: 3D Multi-View MRI Knee Classifier
**Hackathon:** RSNA Knee Abnormality Detection  
**Target:** Kaggle Code Competition  

---

## Phase 1: Exploratory Data Analysis & Preprocessing (Week 1)
- [ ] Inspect DICOM metadata, pixel spacings, slice counts, and sequence distributions across train set.
- [ ] Implement fast DICOM loader with multi-threaded caching using `pydicom` / `SimpleITK`.
- [ ] Build stratified 5-fold cross-validation split grouped by patient ID.
- [ ] Verify voxel normalization and slice interpolation pipeline.

## Phase 2: Baseline Model & Training Loop (Week 2)
- [ ] Implement single-view baseline (Sagittal only) using 2.5D ResNet50 backbone.
- [ ] Setup PyTorch Lightning training loop with mixed precision (AMP) and WandB/TensorBoard logging.
- [ ] Evaluate baseline validation loss and ROC-AUC score.

## Phase 3: Multi-View Fusion & Transformer Backbones (Week 3)
- [ ] Upgrade backbones to Swin Transformer with multi-head attention slice aggregation.
- [ ] Implement multi-sequence late fusion network (Sagittal + Coronal + Axial).
- [ ] Experiment with Asymmetric Focal Loss to penalize rare condition false negatives.
- [ ] Run 5-fold cross-validation training and out-of-fold (OOF) prediction generation.

## Phase 4: Model Ensembling & Kaggle Submission (Week 4)
- [ ] Ensemble top fold models with temperature scaling for probability calibration.
- [ ] Build self-contained Kaggle inference notebook including offline model weight loading.
- [ ] Validate submission file format and verify execution time stays comfortably within the 9-hour limit.
