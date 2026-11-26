"""AI v2-9: patient-level alert replay at frozen v2-8 thresholds.
Comparison half only; no training or Test access. Official utility uses
hourly flags, not cooldown alerts. Onset proxy = first positive label + 6 rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_v2_matched_fpr import patient_split, cutoff_at_fpr, stats, SEED, FPR_TARGETS
from evaluate_utility import utility_components

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports"
SOURCE = REPORT / "v2_ensemble_validation_scores.npz"
V8 = REPORT / "v2_matched_fpr_validation_comparison.json"
MANIFEST = ROOT / "data/splits/patient_splits.json"
INDEX = ROOT / "data/splits/patient_index.csv"
CSV = REPORT / "v2_patient_alert_validation.csv"
JSON = REPORT / "v2_patient_alert_validation.json"
COOLDOWN = 6

def cooldown(flags, hours):
    accepted = []
    for i in np.flatnonzero(flags):
        if not accepted or hours[i] - hours[accepted[-1]] >= COOLDOWN:
            accepted.append(int(i))
    return np.asarray(accepted, dtype=int)

def ratio(a, b):
    return float(a / b) if b else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for path in (SOURCE, V8, MANIFEST, INDEX):
        if not path.is_file():
            parser.error(f"Missing required input: {path}")
    for path in (CSV, JSON):
        if path.exists():
            parser.error(f"Existing output protected: {path}")
    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(groups[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        raise ValueError("Locked split counts changed")
    if len(set(groups["train"] + groups["validation"] + groups["test"])) != 40336:
        raise ValueError("Overlapping patient splits")
    with np.load(SOURCE, allow_pickle=False) as a:
        ids = a["patient_ids"].astype(str)
        times = a["icul_os"].astype(float)
        y = a["labels"].astype(np.int8)
        xgb = a["xgboost_scores"].astype(float)
        v1 = a["lstm_v1_scores"].astype(float)
        v2 = a["lstm_v2_scores"].astype(float)
    if not all(len(a) == 108748 for a in (ids, times, y, xgb, v1, v2)):
        raise ValueError("Unexpected archive lengths")
    if not set(ids).issubset(set(groups["validation"])) or not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid archive IDs or labels")
    if not np.isfinite(times).all():
        raise ValueError("Invalid times")
    for p in (xgb, v1, v2):
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Invalid scores")
    tune, check, nt, nc = patient_split(ids, SEED)
    previous = json.loads(V8.read_text(encoding="utf-8"))
    if (nt, nc) != (2480, 2480) or previous["random_seed"] != SEED:
        raise ValueError("Split differs from v2-8")
    if any(previous[k] != v for k, v in {"tuning_patients": nt, "comparison_patients": nc,
            "tuning_windows": int(tune.sum()), "comparison_windows": int(check.sum()),
            "test_patients_evaluated": 0}.items()):
        raise ValueError("v2-8 scope mismatch")
    scores = {"lstm_v1": v1, "lstm_v2": v2,
              "blend_75_25": .75*v2 + .25*xgb, "blend_90_10": .9*v2 + .1*xgb}
    configs = {}
    for row in previous["rows"]:
        name, fpr = row["model"], row["target_fpr_tuning"]
        key = (name, fpr)
        if key in configs or name not in scores or fpr not in FPR_TARGETS:
            raise ValueError("Unexpected v2-8 operating point")
        threshold = float(row["selected_threshold_tuning_only"])
        if threshold != cutoff_at_fpr(y[tune], scores[name][tune], fpr):
            raise ValueError("Frozen threshold mismatch")
        actual = stats(y[check], scores[name][check], threshold)
        if any(actual[k] != row["comparison_"+k] for k in ("TP","FP","TN","FN")):
            raise ValueError("Frozen comparison counts mismatch")
        configs[key] = threshold
    if len(configs) != 8:
        raise ValueError("Expected eight operating points")
    comparison = set(ids[check])
    index = pd.read_csv(INDEX, dtype={"patient_id": str})
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate index IDs")
    paths = index.set_index("patient_id")["path"]
    if not comparison.issubset(paths.index):
        raise ValueError("Missing comparison patient paths")
    if args.dry_run:
        print("DRY RUN OK: v2-8 split, thresholds and confusion counts reproduced.")
        print(f"Comparison patients: {nc}; Test HELD OUT.")
        print("Output:", CSV, JSON)
        return
    # Group row indices without changing archive order.
    lookup = pd.Series(np.arange(len(ids))).groupby(ids, sort=False).indices
    results = {k: {"raw_flags":0, "alerts":0, "positive_detected":0,
                  "negative_alerted":0, "early_detected":0, "window_detected":0,
                  "lead_hours_proxy":[], "utility":0.0} for k in configs}
    positives = negatives = total_hours = 0
    best_sum = silent_sum = 0.0
    for count, patient in enumerate(sorted(comparison), 1):
        raw = pd.read_csv(Path(paths.loc[patient]), sep="|").sort_values("ICULOS", kind="stable")
        labels = pd.to_numeric(raw["SepsisLabel"], errors="raise").to_numpy()
        hours = pd.to_numeric(raw["ICULOS"], errors="raise").to_numpy(dtype=float)
        idx = lookup[patient]
        if (len(labels) < 24 or not np.isin(labels, (0, 1)).all()
                or not np.isfinite(hours).all() or not np.all(np.diff(hours) == 1)
                or not np.array_equal(labels[23:], y[idx])
                or not np.array_equal(hours[23:], times[idx])):
            raise ValueError(f"Original timeline mismatch: {patient}")
        positive = bool(labels.any())
        positives += int(positive)
        negatives += int(not positive)
        total_hours += len(labels)
        onset = int(np.flatnonzero(labels)[0]) + 6 if positive else None
        alarm, silent, ideal = utility_components(labels)
        best_sum += float(np.where(ideal == 1, alarm, silent).sum())
        silent_sum += float(silent.sum())
        for (name, fpr), threshold in configs.items():
            r = results[name, fpr]
            flags = np.zeros(len(labels), dtype=bool)
            flags[23:] = scores[name][idx] >= threshold
            alerts = cooldown(flags, hours)
            r["raw_flags"] += int(flags.sum())
            r["alerts"] += len(alerts)
            r["utility"] += float(np.where(flags, alarm, silent).sum())
            if positive and len(alerts):
                r["positive_detected"] += 1
                r["lead_hours_proxy"].append(onset - int(alerts[0]))
                r["early_detected"] += int(np.any(alerts <= onset))
                r["window_detected"] += int(np.any((alerts >= onset-12) & (alerts <= onset+3)))
            elif not positive and len(alerts):
                r["negative_alerted"] += 1
        if count % 500 == 0:
            print(f"Alert replay: {count}/{nc}", flush=True)
    denominator = best_sum - silent_sum
    if denominator <= 0 or not positives or not negatives:
        raise ValueError("Invalid comparison cohort normalization")
    rows = []
    for (name, fpr), r in results.items():
        lead = r["lead_hours_proxy"]
        rows.append({"model":name, "target_specificity":1-fpr,
            "frozen_threshold":configs[name,fpr],
            "positive_patients":positives, "negative_patients":negatives,
            "detected_positive_patients":r["positive_detected"],
            "patient_detection_rate":ratio(r["positive_detected"], positives),
            "pre_or_at_onset_detection_rate":ratio(r["early_detected"], positives),
            "utility_window_detection_rate":ratio(r["window_detected"], positives),
            "negative_patients_alerted":r["negative_alerted"],
            "negative_patient_alert_rate":ratio(r["negative_alerted"], negatives),
            "median_first_alert_lead_hours_proxy":float(np.median(lead)) if lead else None,
            "raw_flagged_hours":r["raw_flags"], "cooldown_alerts":r["alerts"],
            "alerts_per_100_patients":100*r["alerts"]/nc,
            "alerts_per_100_patient_days":2400*r["alerts"]/total_hours,
            "normalized_hourly_utility":(r["utility"]-silent_sum)/denominator})
    table = pd.DataFrame(rows).sort_values(["target_specificity","model"], ascending=[False,True])
    table.to_csv(CSV, index=False, mode="x")
    with JSON.open("x", encoding="utf-8") as f:
        json.dump({"scope":"v2-8 comparison half, scoreable patients only",
            "comparison_patients":nc, "positive_patients":positives,
            "negative_patients":negatives, "patient_hours":total_hours,
            "cooldown_hours":COOLDOWN, "test_patients_evaluated":0,
            "limitations":"Previously reused Validation; excludes short stays and tuning half. "
            "Onset proxy follows challenge row convention, not observed clinical diagnosis. "
            "Any-alert detection includes late and very early alerts; window detection is separate. "
            "Lead median is conditional on detected patients. Official utility uses hourly flags, "
            "not cooldown events. Scores are not calibrated probabilities.",
            "rows":rows}, f, indent=2, allow_nan=False)
    print(table[["model","target_specificity","patient_detection_rate",
        "utility_window_detection_rate","negative_patient_alert_rate",
        "median_first_alert_lead_hours_proxy","alerts_per_100_patients",
        "normalized_hourly_utility"]].to_string(index=False))
    print("Saved:", CSV, JSON, "Test HELD OUT.")

if __name__ == "__main__":
    main()
