"""AI-6: held-patient calibration diagnostic on locked Validation score archive.

The original full-Validation model/ensemble selection is already exploratory;
this split reduces direct calibration-fitting leakage but does NOT create an
independent external validation cohort. No Test access; no model training.

From C:\AI:
  python src/training/calibrate_validation_scores.py

Within Validation patients who have >=24 hours, stratified 50/50 patient split:
  calibration subset: fit Platt sigmoid on raw probabilities (logit transform)
  evaluation subset: compare uncalibrated vs calibrated Brier, ECE, AUROC, AP
NO calibrator is saved for deployment to avoid suggesting independent validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

SOURCE = Path("reports/ensemble_validation_scores.npz")
OUT = Path("reports")
EPSILON = 1e-6


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    group = np.searchsorted(np.linspace(0, 1, bins + 1)[1:-1], p, side="right")
    return float(sum(
        (group == i).mean() * abs(float(y[group == i].mean()) -
                                  float(p[group == i].mean()))
        for i in range(bins) if np.any(group == i)
    ))


def calculate(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "mean_score": float(np.mean(p)),
        "Brier": float(brier_score_loss(y, p)),
        "ECE_10_bins": calibration_error(y, p),
        "AUROC": float(roc_auc_score(y, p)),
        "AP": float(average_precision_score(y, p)),
    }


def fit_sigmoid(y: np.ndarray, p: np.ndarray) -> LogisticRegression:
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes required to calibrate")
    logits = np.log(np.clip(p, EPSILON, 1 - EPSILON) /
                    (1 - np.clip(p, EPSILON, 1 - EPSILON)))
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                               random_state=42)
    model.fit(logits.reshape(-1, 1), y)
    return model


def apply_sigmoid(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    logit = np.log(clipped / (1 - clipped))
    return model.predict_proba(logit.reshape(-1, 1))[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.1 <= args.calibration_fraction <= 0.9:
        parser.error("Calibration fraction must be 0.1 to 0.9")

    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing {SOURCE}; run evaluate_ensemble.py")
    with np.load(SOURCE, allow_pickle=False) as data:
        patient = data["patient_ids"].astype(str)
        y = data["labels"].astype(int)
        xgb = data["xgboost_scores"].astype(float)
        lstm = data["lstm_scores"].astype(float)
    if len(y) == 0 or not (len(patient) == len(y) == len(xgb) == len(lstm)):
        raise ValueError("Score archive missing or misaligned")
    if not np.isin(y, (0, 1)).all() or not all(
            np.isfinite(v).all() and np.all((v >= 0) & (v <= 1))
            for v in (xgb, lstm)):
        raise ValueError("Invalid labels/scores")

    manifest = json.loads(Path("data/splits/patient_splits.json").read_text(
        encoding="utf-8"))["split"]
    if not set(patient).issubset(set(manifest["validation"])):
        raise ValueError("Non-Validation patient IDs in score archive")
    unique_patients, inv = np.unique(patient, return_inverse=True)
    # Label strata at patient level, not individual hourly samples.
    patient_positive = np.zeros(len(unique_patients), dtype=int)
    np.maximum.at(patient_positive, inv, y)
    if np.bincount(patient_positive).min() < 2:
        raise ValueError("Not enough positive/negative patients for stratified split")
    indices = np.arange(len(unique_patients))
    fit_indices, check_indices = train_test_split(
        indices, train_size=args.calibration_fraction,
        stratify=patient_positive, random_state=args.seed)
    fit_ids = unique_patients[fit_indices]
    check_ids = unique_patients[check_indices]
    if set(fit_ids) & set(check_ids):
        raise AssertionError("Calibration/evaluation patient leakage")
    fit_mask = np.isin(patient, fit_ids)
    check_mask = np.isin(patient, check_ids)
    if not (np.any(fit_mask) and np.any(check_mask) and
            len(np.unique(y[fit_mask])) == 2 and len(np.unique(y[check_mask])) == 2):
        raise ValueError("Both calibration and evaluation need both label classes")

    models = {
        "xgboost_176": xgb,
        "lstm_full_pw10": lstm,
        "raw_blend_lstm_0.75": 0.25 * xgb + 0.75 * lstm,
    }
    rows = []
    for name, scores in models.items():
        calibrated = fit_sigmoid(y[fit_mask], scores[fit_mask])
        before = calculate(y[check_mask], scores[check_mask])
        after = calculate(y[check_mask],
                          apply_sigmoid(calibrated, scores[check_mask]))
        for status, result in (("uncalibrated", before), ("sigmoid_calibrated", after)):
            rows.append({"model": name, "calibration": status, **result})
        print(f"{name}: Brier {before['Brier']:.5f} -> {after['Brier']:.5f}; "
              f"ECE {before['ECE_10_bins']:.5f} -> {after['ECE_10_bins']:.5f}",
              flush=True)
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / "validation_patient_split_calibration.csv"
    table.to_csv(csv, index=False)
    report = {
        "scope": "fixed_validation_only_with_patient_disjoint_calibration_evaluation",
        "calibration_patients": len(fit_ids),
        "evaluation_patients": len(check_ids),
        "calibration_rows": int(fit_mask.sum()),
        "evaluation_rows": int(check_mask.sum()),
        "total_validation_patients": len(manifest["validation"]),
        "patients_with_scored_windows": len(unique_patients),
        "test_patients_evaluated": 0, "seed": args.seed,
        "fraction": args.calibration_fraction,
        "comparison": rows,
        "limitations": (
            "The underlying XGBoost/LSTM models and 25:75 blend were already "
            "selected using the same fixed Validation cohort, and LSTM early "
            "stopping used full Validation. Patient-disjoint fitting of the "
            "sigmoid calibrator does NOT remove this prior selection leakage. "
            "Diagnostic only; no calibrator persisted for deployment. "
            "Calibrated scores are not guaranteed to generalize."
        ),
    }
    target = OUT / "validation_patient_split_calibration.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"Saved: {csv}, {target}; Test HELD OUT", flush=True)


if __name__ == "__main__":
    main()
