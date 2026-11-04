"""Fast synthetic checks for AI-4 cache; no PhysioNet data needed.

Run: python src/training/test_compare_xgboost.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from compare_xgboost import load_or_build
from train_xgboost import VARS, patient_rows


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "patient.psv"
        data = pd.DataFrame({
            "ICULOS": list(range(1, 27)),
            "SepsisLabel": [0] * 24 + [1, 1],
            **{feature: np.arange(26, dtype=float) for feature in VARS},
        })
        data.loc[4, "HR"] = np.nan
        data.loc[:, "Lactate"] = np.nan
        data.to_csv(path, sep="|", index=False)
        expected, labels = patient_rows(path)
        first = load_or_build("train", ["training_setA/synthetic"], [path], root, False)
        again = load_or_build("train", ["training_setA/synthetic"], [path], root, False)
        for x, y, groups in (first, again):
            assert x.shape == (3, 160), x.shape
            assert np.array_equal(x, np.array(expected), equal_nan=True)
            assert np.array_equal(y, np.array(labels))
            assert groups.tolist() == ["training_setA/synthetic"] * 3
        assert len(list(root.glob("train_*.joblib"))) == 1

        # A changed PSV must not silently use the previous cache.
        data.loc[25, "HR"] = 9999.0
        data.to_csv(path, sep="|", index=False)
        changed = load_or_build("train", ["training_setA/synthetic"], [path], root, False)
        assert len(list(root.glob("train_*.joblib"))) == 2
        assert not np.array_equal(changed[0], first[0], equal_nan=True)
        print("PASS: cached features equal legacy 160 statistics; reuse and invalidation work.")


if __name__ == "__main__":
    main()
