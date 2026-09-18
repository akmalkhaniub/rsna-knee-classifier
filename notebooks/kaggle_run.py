"""RSNA knee Kaggle entry point (baseline).

Discovers study volumes and writes submission.csv. On Kaggle, adapt `load_studies`
to the competition's DICOM layout (see rsnaknee.preprocessing.load_dicom_series).
This baseline uses the heuristic multi-view classifier so it runs with no trained
weights; swap in build_torch_model + loaded weights for a real model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rsnaknee import predict_studies, write_submission  # noqa: E402


def load_studies() -> list[dict]:
    # Placeholder: on Kaggle, iterate the test DICOM dirs and build
    # {"study_id", "sagittal", "coronal", "axial"} preprocessed volumes.
    return []


def main() -> int:
    studies = load_studies()
    if not studies:
        print("No studies found; wire load_studies() to the competition data layout.")
        return 0
    out = "/kaggle/working/submission.csv" if Path("/kaggle/working").exists() else "submission.csv"
    write_submission(predict_studies(studies), out)
    print(f"Wrote {out} for {len(studies)} studies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
