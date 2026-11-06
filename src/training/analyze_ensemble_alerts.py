"""SEWS AI-6: diagnostic alert burden and calibration on locked Validation.

Reads cached aligned XGBoost/LSTM scores from evaluate_ensemble.py.
No model fitting; no Test access; thresholds are PRE-SPECIFIED diagnostics,
not optimized alert cutoffs. Scores are not calibrated sepsis probabilities.
This is NOT the official PhysioNet Challenge 2019 Utility Score.

Run from repository root:
  python src/training/analyze_ensemble_alerts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, average_precision_score, roc_auc_score

SOURCE = Path("reports/ensemble_validation_scores.npz")
OUTPUT = Path("reports")
CUTOFFS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
COOLDOWN_HOURS = 6.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    groups = np.searchsorted(edges[1:-1], p, side="right")
    result = 0.0
    for i in range(bins):
        mask = groups == i
        if np.any(mask):
            result += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def alert_counts(pid: np.ndarray, time: np.ndarray, flagged: np.ndarray,
                 cooldown: float) -> tuple[int, int, int, int]:
    """Simple repeated-alert suppression; NOT a validated clinical alert policy."""
    alerts = 0
    alerted_patients = set()
    eligible = 0
    last_patient = None
    last_alert_time = -np.inf
    for patient, hour, is_flagged in zip(pid, time, flagged):
        if patient != last_patient:
            last_patient = patient
            last_alert_time = -np.inf
        if not is_flagged:
            continue
        eligible += 1
        if float(hour) - last_alert_time >= cooldown:
            alerts += 1
            last_alert_time = float(hour)
            alerted_patients.add(patient)
    return alerts, len(alerted_patients), eligible, len(set(pid))


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Missing aligned scores: {SOURCE}; run evaluate_ensemble.py")
    with np.load(SOURCE, allow_pickle=False) as saved:
        required = {"patient_ids", "icul_os", "labels", "xgboost_scores", "lstm_scores"}
        if not required.issubset(saved.files):
            raise ValueError(f"Missing columns: {sorted(required - set(saved.files))}")
        pid = saved["patient_ids"]
        times = saved["icul_os"].astype(float)
        y = saved["labels"].astype(int)
        xgb = saved["xgboost_scores"].astype(float)
        lstm = saved["lstm_scores"].astype(float)
    n = len(y)
    if not (len(pid) == len(times) == len(xgb) == len(lstm) == n):
        raise ValueError("Row count mismatch")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Nonbinary labels")
    # Some manifest patients may have fewer than 24 observations and thus no scored row.
    manifest = json.loads(Path("data/splits/patient_splits.json").read_text(encoding="utf-8"))["split"]
    validation = set(manifest["validation"])
    if len(validation) != 6454 or not set(pid).issubset(validation):
        raise ValueError("Saved score patient IDs do not match locked Validation")
    if not np.isfinite(times).all():
        raise ValueError("Invalid ICU times")
    for raw in (xgb, lstm):
        if not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
            raise ValueError("Invalid model scores")
    # Patients must occur in contiguous chronological blocks for cooldown logic.
    starts = np.r_[True, pid[1:] != pid[:-1]]
    if len(set(pid[starts])) != int(starts.sum()):
        raise ValueError("Patient rows are not contiguous")
    if np.any((times[1:] < times[:-1]) & ~starts[1:]):
        raise ValueError("Patient ICU times not chronological")
    if np.any((times[1:] == times[:-1]) & ~starts[1:]):
        raise ValueError("Duplicate ICU times within patient")

    scores = {
        "xgboost_176": xgb,
        "lstm_full_pw10": lstm,
        "raw_blend_lstm_0.75": 0.25 * xgb + 0.75 * lstm,
    }
    main_rows, alarm_rows = [], []
    for name, p in scores.items():
        main_rows.append({
            "model": name,
            "rows": n,
            "patients": len(set(pid)),
            "AUROC": float(roc_auc_score(y, p)),
            "AP": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "ece_10_equal_width_bins": expected_calibration_error(y, p),
            "mean_score": float(p.mean()),
            "observed_positive_rate": float(y.mean()),
        })
        for cutoff in CUTOFFS:
            flagged = p >= cutoff
            tp = int(np.count_nonzero(flagged & (y == 1)))
            fp = int(np.count_nonzero(flagged & (y == 0)))
            fn = int(np.count_nonzero(~flagged & (y == 1)))
            tn = int(np.count_nonzero(~flagged & (y == 0)))
            alarms, alerted_patients, flagged_rows, total_patients = alert_counts(
                pid, times, flagged, COOLDOWN_HOURS)
            alarm_rows.append({
                "model": name,
                "score_threshold": cutoff,
                "tp_rows": tp, "fp_rows": fp,
                "fn_rows": fn, "tn_rows": tn,
                "sensitivity_rows": tp / (tp + fn) if tp + fn else None,
                "specificity_rows": tn / (tn + fp) if tn + fp else None,
                "precision_rows": tp / (tp + fp) if tp + fp else None,
                "flagged_rows": flagged_rows,
                "flagged_patients": int(len(set(pid[flagged]))),
                "alerts_6h_cooldown": alarms,
                "patients_with_alerts": alerted_patients,
                "alerts_per_100_validation_patients": 100.0 * alarms / total_patients,
            })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    metric_path = OUTPUT / "ensemble_validation_calibration_diagnostics.csv"
    alert_path = OUTPUT / "ensemble_validation_alert_burden.csv"
    pd.DataFrame(main_rows).to_csv(metric_path, index=False)
    pd.DataFrame(alarm_rows).to_csv(alert_path, index=False)
    report = {
        "scope": "full_locked_validation_only",
        "validation_patients": len(set(pid)),
        "validation_rows": n,
        "test_patients_evaluated": 0,
        "cooldown_hours": COOLDOWN_HOURS,
        "thresholds": CUTOFFS,
        "models": main_rows,
        "warning": (
            "Brier/ECE assess uncalibrated, class-weighted model scores; "
            "fixed thresholds are descriptive, not a clinical cutoff. "
            "The 6-hour cooldown is a simplified simulation without "
            "patient-specific alert adjudication or official utility scoring. "
            "Ensemble selection on the same Validation is optimistic."
        ),
        "files": [str(metric_path), str(alert_path)],
    }
    target = OUTPUT / "ensemble_validation_diagnostics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(pd.DataFrame(main_rows).to_string(index=False), flush=True)
    print("Threshold diagnostics, including 6-hour alert cooldown:")
    print(pd.DataFrame(alarm_rows).to_string(index=False), flush=True)
    print(f"Saved: {metric_path}, {alert_path}, {target}; Test HELD OUT")


if __name__ == "__main__":
    main()
