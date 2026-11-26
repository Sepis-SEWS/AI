"""SEWS AI v2-6: patient-paired bootstrap + descriptive alert burden.

Uses already-aligned original Validation scores from AI v2-5. No fitting or
Test use. Previous outputs are protected and never replaced.
Run:
  python src/training/evaluate_v2_robustness.py --dry-run
  python src/training/evaluate_v2_robustness.py --repeats 200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "v2_ensemble_validation_scores.npz"
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
REPORTS = ROOT / "reports"
BOOT_JSON = REPORTS / "v2_ensemble_patient_bootstrap.json"
BOOT_CSV = REPORTS / "v2_ensemble_patient_bootstrap_repeats.csv"
ALERT_CSV = REPORTS / "v2_ensemble_alert_burden.csv"
ALERT_JSON = REPORTS / "v2_ensemble_alert_diagnostics.json"
CUTOFFS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
COOLDOWN_HOURS = 6.0


def metric(y, score):
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def patient_blocks(pid):
    if not len(pid):
        raise ValueError("No validation rows")
    starts = np.r_[0, np.flatnonzero(pid[1:] != pid[:-1]) + 1]
    stops = np.r_[starts[1:], len(pid)]
    if len(set(pid[starts])) != len(starts):
        raise ValueError("Patient rows are not contiguous")
    return starts, stops


def sample_bootstrap(y, scores, starts, stops, repeats, seed):
    """Whole-patient paired resampling, preserving within-patient temporal dependence."""
    rng = np.random.default_rng(seed)
    n = len(starts)
    records = []
    for iteration in range(repeats):
        chosen = rng.integers(0, n, size=n)
        idx = np.concatenate([np.arange(starts[j], stops[j]) for j in chosen])
        yy = y[idx]
        if len(np.unique(yy)) != 2:
            continue
        auc_lstm, ap_lstm = metric(yy, scores["lstm_v2"][idx])
        auc_blend, ap_blend = metric(yy, scores["blend_75_25"][idx])
        auc_blend90, ap_blend90 = metric(yy, scores["blend_90_10"][idx])
        records.append({
            "repeat": iteration + 1,
            "lstm_AUROC": auc_lstm, "lstm_AP": ap_lstm,
            "blend75_AUROC": auc_blend, "blend75_AP": ap_blend,
            "blend90_AUROC": auc_blend90, "blend90_AP": ap_blend90,
            "blend75_minus_lstm_AUROC": auc_blend - auc_lstm,
            "blend75_minus_lstm_AP": ap_blend - ap_lstm,
            "blend90_minus_lstm_AP": ap_blend90 - ap_lstm,
            "blend75_minus_blend90_AP": ap_blend - ap_blend90,
        })
        if (iteration + 1) % 50 == 0:
            print(f"Patient bootstrap {iteration+1}/{repeats}", flush=True)
    if len(records) < max(20, int(0.9 * repeats)):
        raise ValueError("Too many bootstrap resamples contain only one label")
    return pd.DataFrame(records)


def alert_burden(pid, times, y, scores):
    starts, stops = patient_blocks(pid)
    out = []
    for name, score in scores.items():
        auc, ap = metric(y, score)
        for cutoff in CUTOFFS:
            flagged = score >= cutoff
            tp = int(np.count_nonzero(flagged & (y == 1)))
            fp = int(np.count_nonzero(flagged & (y == 0)))
            fn = int(np.count_nonzero((~flagged) & (y == 1)))
            tn = int(np.count_nonzero((~flagged) & (y == 0)))
            alerts = alerted = 0
            for start, stop in zip(starts, stops):
                last = -np.inf
                any_alert = False
                for hour, is_flagged in zip(times[start:stop], flagged[start:stop]):
                    if is_flagged and hour - last >= COOLDOWN_HOURS:
                        alerts += 1
                        last = hour
                        any_alert = True
                alerted += int(any_alert)
            out.append({
                "model": name, "threshold": cutoff,
                "AUROC": auc, "AP": ap, "TP_rows": tp, "FP_rows": fp,
                "FN_rows": fn, "TN_rows": tn,
                "precision_rows": tp / (tp + fp) if tp + fp else 0.0,
                "recall_rows": tp / (tp + fn) if tp + fn else 0.0,
                "specificity_rows": tn / (tn + fp) if tn + fp else 0.0,
                "flagged_rows": int(flagged.sum()),
                "flagged_patients": int(np.count_nonzero([
                    flagged[a:b].any() for a, b in zip(starts, stops)])),
                "alerts_6h_cooldown": alerts,
                "patients_with_alerts": alerted,
                "alerts_per_100_scored_patients": 100.0 * alerts / len(starts),
            })
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description="SEWS v2 Validation patient bootstrap and alerts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.repeats < 20:
        parser.error("--repeats must be >= 20")
    for needed in (SOURCE, MANIFEST):
        if not needed.is_file():
            parser.error(f"Required existing file missing: {needed}")
    outputs = (BOOT_JSON, BOOT_CSV, ALERT_CSV, ALERT_JSON)
    if args.dry_run:
        print("DRY RUN OK: existing v2 aligned Validation archive and locked manifest found.")
        print("Patient-paired bootstrap repeats:", args.repeats, "seed:", args.seed)
        for p in outputs:
            print("Output:", p, "(EXISTS; won't overwrite)" if p.exists() else "")
        print("No model fitting; Test never accessed.")
        return
    for out in outputs:
        if out.exists():
            parser.error(f"Existing v2 result protected from overwrite: {out}")

    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        raise ValueError("Locked patient split changed")
    if any(set(groups[a]) & set(groups[b])
           for a, b in (("train", "validation"), ("train", "test"),
                        ("validation", "test"))):
        raise ValueError("Overlapping locked patient IDs")
    with np.load(SOURCE, allow_pickle=False) as data:
        required = {"patient_ids", "icul_os", "labels", "xgboost_scores",
                    "lstm_v1_scores", "lstm_v2_scores"}
        if not required.issubset(set(data.files)):
            raise ValueError("v2-5 aligned archive missing fields")
        pid = data["patient_ids"].astype(str)
        times = data["icul_os"].astype(float)
        y = data["labels"].astype(np.int8)
        xgb = data["xgboost_scores"].astype(np.float64)
        lstm_v1 = data["lstm_v1_scores"].astype(np.float64)
        lstm = data["lstm_v2_scores"].astype(np.float64)
    if not (len(pid) == len(times) == len(y) == len(xgb) == len(lstm_v1) == len(lstm)):
        raise ValueError("Archive lengths differ")
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("Invalid Validation labels")
    if not set(pid).issubset(set(groups["validation"])):
        raise ValueError("Non-Validation patient in archive")
    for score in (xgb, lstm_v1, lstm):
        if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("Nonfinite or out-of-range model score")
    if not np.isfinite(times).all():
        raise ValueError("Invalid ICU time")
    starts, stops = patient_blocks(pid)
    for start, stop in zip(starts, stops):
        if np.any(np.diff(times[start:stop]) <= 0):
            raise ValueError("Patient ICU time must be strictly increasing")

    scores = {
        "xgboost_v1": xgb,
        "lstm_v1": lstm_v1,
        "lstm_v2": lstm,
        "blend_75_25": 0.75 * lstm + 0.25 * xgb,
        "blend_90_10": 0.90 * lstm + 0.10 * xgb,
    }
    point = {k: {"AUROC": metric(y, score)[0], "AP": metric(y, score)[1]}
             for k, score in scores.items()}
    print("Validation patients with scoreable windows:", len(starts), "rows:", len(y))
    print("Point estimates:", json.dumps(point, indent=2), flush=True)
    frame = sample_bootstrap(y, scores, starts, stops, args.repeats, args.seed)
    deltas = {}
    definitions = {
        "blend75_minus_lstm_AP": ("blend_75_25", "lstm_v2", "AP"),
        "blend75_minus_lstm_AUROC": ("blend_75_25", "lstm_v2", "AUROC"),
        "blend90_minus_lstm_AP": ("blend_90_10", "lstm_v2", "AP"),
        "blend75_minus_blend90_AP": ("blend_75_25", "blend_90_10", "AP"),
    }
    for name, (a, b, key) in definitions.items():
        values = frame[name].to_numpy()
        deltas[name] = {
            "point_delta": point[a][key] - point[b][key],
            "bootstrap_mean": float(values.mean()),
            "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
            "fraction_le_zero": float(np.mean(values <= 0)),
        }
    alarms = alert_burden(pid, times, y, scores)
    BOOT_JSON.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(BOOT_CSV, index=False)
    BOOT_JSON.write_text(json.dumps({
        "scope": "locked_validation_scored_patients_only",
        "total_validation_patients": 6454, "scored_patients": len(starts),
        "scoreable_rows": len(y), "repeats_requested": args.repeats,
        "repeats_valid": len(frame), "seed": args.seed,
        "test_patients_evaluated": 0, "point_metrics": point,
        "paired_deltas": deltas,
        "warning": (
            "Whole-patient paired resampling within a previously repeatedly used "
            "Validation cohort. Intervals quantify sampling variation, NOT "
            "independent out-of-sample performance or corrected weight-selection "
            "bias. No model refitting and no Test access.")
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    alarms.to_csv(ALERT_CSV, index=False)
    ALERT_JSON.write_text(json.dumps({
        "scope": "locked_validation_scoreable_rows_only",
        "test_patients_evaluated": 0,
        "thresholds": CUTOFFS, "cooldown_hours": COOLDOWN_HOURS,
        "total_validation_patients": 6454,
        "scored_patients": len(starts),
        "scoreable_rows": len(y),
        "warning": (
            "Fixed diagnostic score thresholds, NOT tuned clinical cutoffs. "
            "Weighted-model scores are not calibrated risks. "
            "6-hour suppression is a simplified alert count, NOT the official "
            "PhysioNet 2019 Utility metric. Rates use scored patients only. "
            "No predictions at the first 23 hours and no patient-level onset "
            "adjudication in these alert diagnostics."),
        "rows": alarms.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Paired bootstrap:", json.dumps(deltas, indent=2), flush=True)
    print("Alert diagnostic preview, threshold=0.2:")
    print(alarms.loc[alarms.threshold == 0.2, [
        "model", "precision_rows", "recall_rows", "FP_rows",
        "alerts_per_100_scored_patients"]].to_string(index=False), flush=True)
    print("Saved:", BOOT_JSON, BOOT_CSV, ALERT_JSON, ALERT_CSV,
          "; Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
