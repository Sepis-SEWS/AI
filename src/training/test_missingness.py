"""Synthetic causal feature checks for experiment_missingness.py.

Run: python src/training/test_missingness.py
"""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from experiment_missingness import EXTRA_NAMES, patient_missingness, chunks_for
from train_xgboost import VARS


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        path = root / "sample.psv"
        data = pd.DataFrame({
            "ICULOS": range(1, 27),
            "SepsisLabel": [0] * 25 + [1],
            **{name: np.full(26, np.nan) for name in VARS},
        })
        data.loc[0, "HR"] = 75.
        data.loc[23, "HR"] = 99.
        data.loc[25, "HR"] = 101.
        data.to_csv(path, index=False, sep="|")
        first = patient_missingness(path)
        assert first.shape == (3, 48)
        f = {name: i for i, name in enumerate(EXTRA_NAMES)}
        assert first[0, f["HR_observed_24h"]] == 2
        assert first[0, f["HR_observed_6h"]] == 1
        assert first[0, f["HR_age_hours"]] == 0
        assert first[1, f["HR_observed_24h"]] == 1
        assert first[1, f["HR_age_hours"]] == 1
        assert first[2, f["HR_age_hours"]] == 0
        assert first[0, f["Lactate_observed_24h"]] == 0
        assert first[0, f["Lactate_age_hours"]] == 48

        # Future values must not affect earlier windows.
        changed = data.copy()
        changed.loc[25, "Lactate"] = 3.
        changed.to_csv(path, index=False, sep="|")
        second = patient_missingness(path)
        assert np.array_equal(first[:2], second[:2])
        assert second[2, f["Lactate_observed_24h"]] == 1
        checkpoint = chunks_for("train", ["training_setA/sample"], [path], root, False)
        assert np.array_equal(checkpoint, second)
        checkpoint2 = chunks_for("train", ["training_setA/sample"], [path], root, False)
        assert np.array_equal(checkpoint, checkpoint2)
    print("PASS: 48 causal features, temporal isolation, and checkpoint resume")


if __name__ == "__main__":
    main()
