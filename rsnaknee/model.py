"""Knee-abnormality classifiers.

- ``HeuristicMultiViewClassifier``: a no-train baseline that fuses per-sequence signal
  into calibrated probabilities (ported from the prototype; real NumPy).
- ``build_torch_model``: a 2.5D multi-view fusion network scaffold, created only when
  PyTorch is installed so the package stays importable without it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LABELS = ("acl_tear", "meniscus_tear", "cartilage_abnormality")


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class StudyPrediction:
    study_id: str
    probabilities: dict[str, float]

    @property
    def impression(self) -> dict[str, bool]:
        return {k: v > 0.5 for k, v in self.probabilities.items()}


class HeuristicMultiViewClassifier:
    """Cross-view fusion baseline over sagittal/coronal/axial mean signal."""

    def _seq_mean(self, vol) -> float:
        if vol is None:
            return 0.5
        arr = np.asarray(vol, dtype=np.float32)
        return float(arr.mean()) if arr.size else 0.5

    def predict(self, study: dict) -> StudyPrediction:
        sag = self._seq_mean(study.get("sagittal"))
        cor = self._seq_mean(study.get("coronal"))
        axi = self._seq_mean(study.get("axial"))
        p_acl = float(sigmoid(sag * 2.4 + cor * 0.8 - 1.5))
        p_men = float(sigmoid(cor * 2.2 + sag * 1.1 - 1.2))
        p_car = float(sigmoid(axi * 2.0 + cor * 0.9 - 1.4))
        return StudyPrediction(
            study_id=str(study.get("study_id", "unknown")),
            probabilities={
                "acl_tear": round(p_acl, 4),
                "meniscus_tear": round(p_men, 4),
                "cartilage_abnormality": round(p_car, 4),
            },
        )


def build_torch_model(num_labels: int = len(LABELS)):  # pragma: no cover - optional dep
    """Build a 2.5D multi-view fusion model (requires torch + torchvision)."""
    try:
        import torch
        import torch.nn as nn
        from torchvision.models import resnet18
    except ImportError as exc:
        raise RuntimeError("PyTorch + torchvision required for build_torch_model.") from exc

    class MultiViewNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            def backbone():
                m = resnet18(weights=None)
                m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # single-channel MRI slice
                m.fc = nn.Identity()
                return m

            self.sag, self.cor, self.axi = backbone(), backbone(), backbone()
            self.head = nn.Sequential(nn.Linear(512 * 3, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_labels))

        def forward(self, sag, cor, axi):  # each (B, 1, H, W): the center-slice of the 2.5D stack
            feats = torch.cat([self.sag(sag), self.cor(cor), self.axi(axi)], dim=1)
            return self.head(feats)

    return MultiViewNet()
