"""SEWS AI v2-8: compare models at the same FALSE-POSITIVE rate.

Split existing locked Validation patients into two non-overlapping patient
groups. Choose each model's operating threshold on one group to aim at 95%
or 90% specificity, then evaluate that frozen threshold on the other group.
The models/weights were previously compared on the FULL Validation cohort,
so this is an internal diagnostic, NOT an untouched independent validation.
No training, model fitting, Test access, or modification of existing files.

Run from repository root:
  python src/training/evaluate_v2_matched_fpr.py --dry-run
  python src/training/evaluate_v2_matched_fpr.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "reports" / "v2_ensemble_validation_scores.npz"
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
OUT_CSV = ROOT / "reports" / "v2_matched_fpr_validation_comparison.csv"
OUT_JSON = ROOT / "reports" / "v2_matched_fpr_validation_comparison.json"
FPR_TARGETS = (0.05, 0.10)
SEED = 42


def patient_split(ids, seed):
    """Exact deterministic whole-patient partition by manifest ordering."""
    unique, first = np.unique(ids, return_index=True)
    unique = unique[np.argsort(first)]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique))
    tune = set(unique[perm[:len(unique) // 2]])
    check = set(unique[perm[len(unique) // 2:]])
    if tune & check or len(tune) + len(check) != len(unique):
        raise ValueError("Patient split was not disjoint")
    return np.isin(ids, list(tune)), np.isin(ids, list(check)), len(tune), len(check)


def cutoff_at_fpr(y, score, max_fpr):
    """Select a conservative score cutoff with FPR no greater than budget.

    Negatives >= cutoff count as false positives. Start just above a negative
    score at the allowed rank; this handles ties deterministically.
    """
    negative = np.sort(score[y == 0])
    if not len(negative) or not np.any(y == 1):
        raise ValueError("Both classes required for calibration")
    allowed = int(np.floor(max_fpr * len(negative)))
    if allowed == 0:
        return float(np.nextafter(negative[-1], np.inf))
    cutoff = float(np.nextafter(negative[-allowed - 1], np.inf))
    if not np.isfinite(cutoff):
        raise ValueError("Selected invalid threshold")
    if np.mean(negative >= cutoff) > max_fpr + 1e-12:
        raise AssertionError("FPR calibration failed")
    return cutoff


def stats(y, p, threshold):
    flagged = p >= threshold
    positive = y == 1
    tp = int(np.count_nonzero(flagged & positive))
    fp = int(np.count_nonzero(flagged & ~positive))
    fn = int(np.count_nonzero(~flagged & positive))
    tn = int(np.count_nonzero(~flagged & ~positive))
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (tn + fp) if tn + fp else 0.0,
        "flagged_rows": tp + fp,
    }


def main():
    parser = argparse.ArgumentParser(description="V2 candidate comparison at matched FPR")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for p in (INPUT, MANIFEST):
        if not p.is_file():
            parser.error(f"Missing required input: {p}")
    if args.dry_run:
        print("DRY RUN OK: Validation archive and locked manifest present.")
        for path in (OUT_CSV, OUT_JSON):
            print("Output:", path, "(already exists; protected)" if path.exists() else "")
        print("Patient-disjoint tuning/comparison groups. No Test access or retraining.")
        return
    for path in (OUT_CSV, OUT_JSON):
        if path.exists():
            parser.error(f"Existing output protected: {path}")
    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        raise ValueError("Locked manifest changed")
    if len(set(groups["train"] + groups["validation"] + groups["test"])) != 40336:
        raise ValueError("Overlapping patient splits")

    with np.load(INPUT, allow_pickle=False) as archive:
        expected = {"patient_ids", "labels", "xgboost_scores",
                    "lstm_v1_scores", "lstm_v2_scores"}
        if not expected.issubset(archive.files):
            raise ValueError("Incomplete v2 Validation archive")
        pid = archive["patient_ids"].astype(str)
        y = archive["labels"].astype(np.int8)
        xgb = archive["xgboost_scores"].astype(float)
        v1 = archive["lstm_v1_scores"].astype(float)
        v2 = archive["lstm_v2_scores"].astype(float)
    if not (len(y) == len(pid) == len(xgb) == len(v1) == len(v2) == 108748):
        raise ValueError("Unexpected Validation row count")
    if not set(pid).issubset(set(groups["validation"])):
        raise ValueError("Archive includes non-Validation patient")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Nonbinary target labels")
    for p in (xgb, v1, v2):
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Invalid scores")
    # Split only patients with scoreable 24-hour windows. Other Validation
    # patients remain excluded from these window-level operating metrics.
    tune, check, n_tune, n_check = patient_split(pid, SEED)
    if np.any(tune & check) or not np.all(tune | check):
        raise ValueError("Split not exhaustive")
    if len(np.unique(y[tune])) != 2 or len(np.unique(y[check])) != 2:
        raise ValueError("Both groups need positive and negative windows")

    scores = {
        "lstm_v1": v1, "lstm_v2": v2,
        "blend_75_25": 0.75 * v2 + 0.25 * xgb,
        "blend_90_10": 0.90 * v2 + 0.10 * xgb,
    }
    rows = []
    for target in FPR_TARGETS:
        for name, p in scores.items():
            threshold = cutoff_at_fpr(y[tune], p[tune], target)
            cal = stats(y[tune], p[tune], threshold)
            report = stats(y[check], p[check], threshold)
            rows.append({
                "model": name,
                "target_fpr_tuning": target,
                "target_specificity_tuning": 1.0 - target,
                "selected_threshold_tuning_only": threshold,
                "tuning_FPR": cal["false_positive_rate"],
                "comparison_FPR": report["false_positive_rate"],
                "comparison_specificity": report["specificity"],
                "comparison_precision": report["precision"],
                "comparison_recall": report["recall"],
                "comparison_TP": report["TP"], "comparison_FP": report["FP"],
                "comparison_FN": report["FN"], "comparison_TN": report["TN"],
                "comparison_AUROC": float(roc_auc_score(y[check], p[check])),
                "comparison_AP": float(average_precision_score(y[check], p[check])),
            })
    table = pd.DataFrame(rows).sort_values(["target_fpr_tuning", "model"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({
        "scope": "locked_validation_patient_disjoint_internal_diagnostic",
        "total_validation_patients": 6454,
        "scored_validation_patients": n_tune + n_check,
        "tuning_patients": n_tune, "comparison_patients": n_check,
        "tuning_windows": int(tune.sum()), "comparison_windows": int(check.sum()),
        "random_seed": SEED, "target_false_positive_rates": FPR_TARGETS,
        "test_patients_evaluated": 0,
        "rows": rows,
        "critical_limitation": (
            "Thresholds are fit on a patient-disjoint subset, but the candidate "
            "models and blend weights were previously selected using the ENTIRE "
            "same Validation cohort. Therefore the comparison half is NOT a "
            "previously unseen independent holdout and selection bias persists. "
            "Differences in observed comparison FPR are expected, so verify "
            "specificity before asserting fair equal-FPR comparison. Scores "
            "are uncalibrated. No new Train or Test evaluation."),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Patient-split: tuning {n_tune}, comparison {n_check}; Test HELD OUT.")
    print(table[["model", "target_specificity_tuning", "selected_threshold_tuning_only",
                 "comparison_specificity", "comparison_precision", "comparison_recall",
                 "comparison_FP", "comparison_AP"]].to_string(index=False), flush=True)
    print("Saved:", OUT_CSV, OUT_JSON, "Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
