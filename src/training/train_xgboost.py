"""SEWS XGBoost baseline: patient-wise split, causal 24-hour features.

Smoke test:
  python src/training/train_xgboost.py --max-patients 200
Full first baseline:
  python src/training/train_xgboost.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

VARS = [
    "HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total",
    "Platelets", "FiO2", "BUN", "Glucose", "pH",
]
FEATURES = [
    f"{name}_{stat}"
    for name in VARS
    for stat in ("mean", "std", "min", "max", "median", "iqr", "skew",
                 "slope", "rate", "cv")
]
ROOT = Path("data/data-sepsis/training")


def file_list(max_patients: int | None) -> list[Path]:
    """List Set A/B patient files and optionally restrict smoke-test size."""
    files = sorted(ROOT.glob("training_setA/*.psv"))
    files += sorted(ROOT.glob("training_setB/*.psv"))
    if max_patients is not None:
        files = files[:max_patients]
    if not files:
        raise FileNotFoundError(f"No PSV files under {ROOT.resolve()}")
    return files


def patient_rows(path: Path) -> tuple[list[list[float]], list[int]]:
    """Create features using only the current and previous 23 hourly rows."""
    data = pd.read_csv(path, sep="|")
    expected = set(VARS) | {"SepsisLabel", "ICULOS"}
    missing = expected - set(data.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    data = data.sort_values("ICULOS", kind="stable").reset_index(drop=True)
    data[VARS] = data[VARS].apply(pd.to_numeric, errors="coerce")
    rows: list[list[float]] = []
    targets: list[int] = []
    # Start at hour 24, where a full 24-row history exists.
    for end in range(23, len(data)):
        current = data.iloc[end]
        block = data.iloc[end - 23:end + 1]
        vector: list[float] = []
        for name in VARS:
            arr = block[name].to_numpy(dtype=float)
            times = block["ICULOS"].to_numpy(dtype=float)
            good = np.isfinite(arr) & np.isfinite(times)
            vals = arr[good]
            t = times[good]
            if len(vals) == 0:
                vector.extend([np.nan] * 10)
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals)) if len(vals) >= 2 else np.nan
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            median = float(np.median(vals))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            # Undefined for fewer than three observations / constant values.
            if len(vals) >= 3 and np.std(vals) > 1e-12:
                skew = float(np.mean(((vals - mean) / np.std(vals)) ** 3))
            else:
                skew = np.nan
            if len(vals) >= 2 and np.ptp(t) > 0:
                slope = float(np.polyfit(t - t[0], vals, 1)[0])
                rate = float(np.mean(np.diff(vals) / np.diff(t))) if np.all(np.diff(t) > 0) else np.nan
            else:
                slope, rate = np.nan, np.nan
            cv = float(std / abs(mean)) if np.isfinite(std) and abs(mean) > 1e-12 else np.nan
            vector.extend([mean, std, vmin, vmax, median, iqr,
                           skew, slope, rate, cv])
        rows.append(vector)
        targets.append(int(current["SepsisLabel"]))
    return rows, targets


def load_group(paths: list[Path]) -> tuple[pd.DataFrame, np.ndarray]:
    """Assemble per-hour samples from a disjoint group of patient files."""
    all_x: list[list[float]] = []
    all_y: list[int] = []
    for index, path in enumerate(paths, start=1):
        x, y = patient_rows(path)
        all_x.extend(x)
        all_y.extend(y)
        if index % 1000 == 0:
            print(f"  read {index}/{len(paths)} patients")
    return pd.DataFrame(all_x, columns=FEATURES), np.array(all_y, dtype=int)


def score(name: str, y: np.ndarray, prob: np.ndarray) -> None:
    """Print metrics, protecting against one-class holdouts."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        print(f"{name}: only one label present, AUROC/AUPRC unavailable")
        return
    print(f"{name}: AUROC={roc_auc_score(y, prob):.4f} "
          f"AUPRC={average_precision_score(y, prob):.4f} "
          f"positive_rate={np.mean(y):.4f}")


def main() -> None:
    """Train a simple first baseline without using test data for fitting."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Use first N patient files for a quick smoke test")
    args = parser.parse_args()

    paths = file_list(args.max_patients)
    train_dev, test = train_test_split(paths, test_size=0.2, random_state=42)
    train, val = train_test_split(train_dev, test_size=0.2, random_state=42)
    print(f"Patients: train={len(train)} val={len(val)} test={len(test)}")
    print("Generating causal 24-hour statistics (160 features)...")
    X_train, y_train = load_group(train)
    X_val, y_val = load_group(val)
    print(f"Training samples={len(y_train)}; positive={int(y_train.sum())}")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Train has only one class. Increase --max-patients.")
    positive = int(y_train.sum())
    negative = len(y_train) - positive
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=negative / positive, eval_metric="logloss",
        n_jobs=-1, random_state=42,
    )
    print("Training XGBoost...")
    model.fit(X_train, y_train)
    score("Validation", y_val, model.predict_proba(X_val)[:, 1])
    out = Path("models")
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES,
                 "selected_columns": VARS, "window_hours": 24},
                out / "xgboost_baseline.joblib")
    print("Saved models/xgboost_baseline.joblib")
    print("Test held out; final scoring, 5-fold CV, Utility and calibration NOT yet implemented.")


if __name__ == "__main__":
    main()
