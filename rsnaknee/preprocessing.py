"""Volumetric MRI preprocessing: DICOM loading, depth standardization, normalization.

All array math is real NumPy. DICOM reading uses ``pydicom`` when available; the rest
works on plain arrays so it is testable without medical data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def standardize_depth(volume: np.ndarray, target_depth: int = 32) -> np.ndarray:
    """Resample a (D, H, W) volume to ``target_depth`` slices by nearest-index sampling."""
    volume = np.asarray(volume, dtype=np.float32)
    d = volume.shape[0]
    if d == target_depth:
        return volume
    if d == 1:
        return np.repeat(volume, target_depth, axis=0)
    idx = np.round(np.linspace(0, d - 1, target_depth)).astype(int)
    return volume[idx]


def normalize_intensity(volume: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-clip then z-score normalize voxel intensities."""
    volume = np.asarray(volume, dtype=np.float32)
    lo, hi = np.percentile(volume, [p_low, p_high])
    clipped = np.clip(volume, lo, hi)
    mean = float(clipped.mean())
    std = float(clipped.std()) or 1.0
    return (clipped - mean) / std


def preprocess_volume(volume: np.ndarray, target_depth: int = 32) -> np.ndarray:
    return normalize_intensity(standardize_depth(volume, target_depth))


def load_dicom_series(directory: str | Path) -> np.ndarray:
    """Load a DICOM series into a (D, H, W) float32 volume, ordered by InstanceNumber.

    Requires ``pydicom``. Raises a clear error if unavailable.
    """
    try:
        import pydicom  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("pydicom is required to load DICOM series (pip install pydicom).") from exc

    files = sorted(Path(directory).glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files in {directory}")
    datasets = [pydicom.dcmread(str(f)) for f in files]
    datasets.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
    return np.stack([ds.pixel_array.astype(np.float32) for ds in datasets], axis=0)
