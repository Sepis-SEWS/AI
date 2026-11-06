"""SEWS AI-6 paired PATIENT bootstrap: is ensemble lift vs LSTM robust?

Reuses locked Validation score archive. Samples PATIENTS with replacement,
never individual hourly rows (which are correlated). Compares AUROC/AP
of LSTM, XGBoost, 25:75 raw blend. Uses 95% percentile intervals of
paired deltas, not independent test confidence intervals.

Usage:
    python src/training/bootstrap_ensemble.py --repeats 200
    python src/training/bootstrap_ensemble.py --repeats 500 --seed 42

No fitting, no Test, no changes to model or patient split.
Selected blend and thresholds have already been compared on this same
Validation set, so uncertainty intervals do not correct selection bias.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

INPUT = Path("reports/ensemble_validation_scores.npz")
OUTPUT = Path("reports")


def metric_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def grouped_indices(patient_ids: np.ndarray) -> list[np.ndarray]:
    """Return contiguous per-patient slices, validate no repeated disjoint blocks."""
    start = np.r_[0, np.flatnonzero(patient_ids[1:] != patient_ids[:-1]) + 1]
    stop = np.r_[start[1:], len(patient_ids)]
    ids = patient_ids[start]
    if len(ids) != len(set(ids)):
        raise ValueError("Non-contiguous rows for at least one patient")
    return [np.arange(a, b, dtype=np.int64) for a, b in zip(start, stop)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 20:
        parser.error("Use at least 20 bootstrap repeats")
    if not INPUT.is_file():
        raise FileNotFoundError(f"Missing {INPUT}: run evaluate_ensemble.py")
    with np.load(INPUT, allow_pickle=False) as archive:
        pid = archive["patient_ids"]
        labels = archive["labels"].astype(np.int8)
        xgb = archive["xgboost_scores"].astype(np.float64)
        lstm = archive["lstm_scores"].astype(np.float64)
    if not (len(pid) == len(labels) == len(xgb) == len(lstm)):
        raise ValueError("Input arrays have mismatched lengths")
    if len(labels) == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Missing/invalid binary labels")
    if not (np.isfinite(xgb).all() and np.isfinite(lstm).all()):
        raise ValueError("Non-finite scores")
    manifest = json.loads(Path("data/splits/patient_splits.json").read_text(
        encoding="utf-8"))["split"]
    if not set(pid).issubset(set(manifest["validation"])):
        raise ValueError("Non-Validation patient in score archive")
    blocks = grouped_indices(pid)
    count = len(blocks)
    if count < 2:
        raise ValueError("Too few scored patients")
    scores = {
        "xgboost_176": xgb,
        "lstm_full_pw10": lstm,
        "raw_blend_lstm_0.75": 0.25 * xgb + 0.75 * lstm,
    }
    point = {name: metric_pair(labels, values) for name, values in scores.items()}
    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    for repeat in range(args.repeats):
        chosen = rng.integers(0, count, size=count)
        selected = np.concatenate([blocks[j] for j in chosen])
        yy = labels[selected]
        if len(np.unique(yy)) != 2:
            # Rare in small datasets; do not silently include invalid bootstrap.
            continue
        x_auc, x_ap = metric_pair(yy, xgb[selected])
        l_auc, l_ap = metric_pair(yy, lstm[selected])
        b_auc, b_ap = metric_pair(yy, scores["raw_blend_lstm_0.75"][selected])
        records.append({
            "repeat": repeat,
            "xgboost_AUROC": x_auc, "xgboost_AP": x_ap,
            "lstm_AUROC": l_auc, "lstm_AP": l_ap,
            "blend_AUROC": b_auc, "blend_AP": b_ap,
            "blend_minus_lstm_AUROC": b_auc - l_auc,
            "blend_minus_lstm_AP": b_ap - l_ap,
        })
        if (repeat + 1) % 50 == 0:
            print(f"Patient bootstrap {repeat + 1}/{args.repeats}", flush=True)
    if len(records) < max(20, args.repeats * 0.9):
        raise ValueError("Too many resamples had only one label class")
    frame = pd.DataFrame(records)
    outcome = {}
    for measure in ("AUROC", "AP"):
        field = f"blend_minus_lstm_{measure}"
        vals = frame[field].to_numpy()
        outcome[field] = {
            "point_delta": point["raw_blend_lstm_0.75"][0 if measure == "AUROC" else 1]
                           - point["lstm_full_pw10"][0 if measure == "AUROC" else 1],
            "bootstrap_mean": float(vals.mean()),
            "percentile_95_CI": np.quantile(vals, [0.025, 0.975]).tolist(),
            "fraction_delta_le_zero": float(np.mean(vals <= 0)),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = OUTPUT / "ensemble_patient_bootstrap_repeats.csv"
    report_file = OUTPUT / "ensemble_patient_bootstrap.json"
    frame.to_csv(rows, index=False)
    report = {
        "scope": "locked_validation_scored_patients_only",
        "patients_with_24row_windows": count,
        "total_validation_patients": len(manifest["validation"]),
        "hourly_rows": len(labels),
        "repeats_requested": args.repeats,
        "repeats_valid": len(records),
        "seed": args.seed, "test_patients_evaluated": 0,
        "point_estimates": {
            k: {"AUROC": v[0], "AP": v[1]} for k, v in point.items()
        },
        "paired_deltas_vs_lstm": outcome,
        "notes": (
            "Resamples whole patients with replacement; all 24-row-scored hours "
            "for sampled patients remain together. Intervals quantify sampling "
            "variation on an already-used fixed Validation set, not independent "
            "generalization nor model-selection uncertainty. All blend weights "
            "were previously evaluated on Validation. No clinical claims."
        ),
    }
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps(outcome, indent=2), flush=True)
    print(f"Saved: {report_file}, {rows}. Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
