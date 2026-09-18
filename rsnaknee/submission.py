"""Build and validate an RSNA knee submission CSV.

Format: one row per study, a column per abnormality label with a probability in [0, 1].
Uses stdlib csv (no pandas dependency).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .model import LABELS, HeuristicMultiViewClassifier, StudyPrediction


def predict_studies(studies: Iterable[dict], classifier: HeuristicMultiViewClassifier | None = None) -> list[StudyPrediction]:
    clf = classifier or HeuristicMultiViewClassifier()
    return [clf.predict(s) for s in studies]


def write_submission(preds: list[StudyPrediction], out_path: str | Path = "submission.csv") -> Path:
    out = Path(out_path)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["study_id", *LABELS])
        for p in preds:
            writer.writerow([p.study_id, *[f"{p.probabilities[l]:.4f}" for l in LABELS]])
    return out


def validate_submission_rows(rows: list[dict]) -> None:
    """Raise if any row is malformed (missing label or probability out of range)."""
    for row in rows:
        assert "study_id" in row, "row missing study_id"
        for label in LABELS:
            assert label in row, f"row missing label {label}"
            val = float(row[label])
            assert 0.0 <= val <= 1.0, f"{label}={val} out of [0,1]"


def read_submission(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as fh:
        return list(csv.DictReader(fh))
