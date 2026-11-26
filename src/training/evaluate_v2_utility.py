"""SEWS AI v2-7: PhysioNet/CinC 2019 normalized Utility on locked Validation.

Reuses the already aligned v2-5 score archive. Does NOT retrain models, read
Test patient data, change the v1 locked policy, or overwrite prior reports.

Full original patient timelines are evaluated, including the first 23 hours
with no possible LSTM prediction (forced 0). Uses the verified existing
utility_components() implementation from evaluate_utility.py.
Run:
    python src/training/evaluate_v2_utility.py --dry-run
    python src/training/evaluate_v2_utility.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_utility import utility_components

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "reports" / "v2_ensemble_validation_scores.npz"
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
INDEX = ROOT / "data" / "splits" / "patient_index.csv"
CSV_OUTPUT = ROOT / "reports" / "v2_ensemble_validation_utility_by_threshold.csv"
JSON_OUTPUT = ROOT / "reports" / "v2_ensemble_validation_utility.json"
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
MODELS = ("lstm_v1", "lstm_v2", "blend_75_25", "blend_90_10")


def main():
    parser = argparse.ArgumentParser(description="V2 official-formula Validation utility (no Test)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (SOURCE, MANIFEST, INDEX):
        if not path.is_file():
            parser.error(f"Missing required input: {path}")
    if args.dry_run:
        print("DRY RUN OK: archived v2 Validation scores, manifest, and index exist.")
        print("Validation patient timelines only; first 23 hours silent; Test held out.")
        for p in (CSV_OUTPUT, JSON_OUTPUT):
            print("Output:", p, "(EXISTS: protected)" if p.exists() else "")
        return
    for p in (CSV_OUTPUT, JSON_OUTPUT):
        if p.exists():
            parser.error(f"Existing result protected from overwrite: {p}")

    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    val_ids = groups["validation"]
    if tuple(len(groups[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        raise ValueError("Locked split counts differ")
    if len(set(groups["train"] + val_ids + groups["test"])) != 40336:
        raise ValueError("Patient IDs overlap across locked splits")
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in patient index")
    paths = index.set_index("patient_id")["path"]
    if not set(val_ids).issubset(paths.index):
        raise ValueError("Missing Validation patient in index")

    with np.load(SOURCE, allow_pickle=False) as data:
        needed = {"patient_ids", "icul_os", "labels", "xgboost_scores",
                  "lstm_v1_scores", "lstm_v2_scores"}
        if not needed.issubset(data.files):
            raise ValueError("Missing score fields in v2 archive")
        ids = data["patient_ids"].astype(str)
        hours = data["icul_os"].astype(float)
        labels = data["labels"].astype(np.int8)
        xgb = data["xgboost_scores"].astype(np.float64)
        v1 = data["lstm_v1_scores"].astype(np.float64)
        v2 = data["lstm_v2_scores"].astype(np.float64)

    n = len(labels)
    if not (len(ids) == len(hours) == len(xgb) == len(v1) == len(v2) == n):
        raise ValueError("Score arrays differ in length")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(hours).all():
        raise ValueError("Nonbinary labels or invalid ICU timestamps")
    if not set(ids).issubset(set(val_ids)):
        raise ValueError("Score archive contains non-Validation patients")
    for values in (xgb, v1, v2):
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Model score out of bounds")

    scores = {
        "lstm_v1": v1,
        "lstm_v2": v2,
        "blend_75_25": 0.75 * v2 + 0.25 * xgb,
        "blend_90_10": 0.9 * v2 + 0.1 * xgb,
    }
    utility_sums = {(model, t): 0.0 for model in MODELS for t in THRESHOLDS}
    best_sum = inactive_sum = 0.0
    cursor = short = scoreable = 0
    total_hours = 0

    for count, patient_id in enumerate(val_ids, start=1):
        raw = pd.read_csv(Path(paths.loc[patient_id]), sep="|")
        raw = raw.sort_values("ICULOS", kind="stable").reset_index(drop=True)
        original_y = pd.to_numeric(raw["SepsisLabel"], errors="raise").to_numpy(dtype=np.int8)
        original_t = pd.to_numeric(raw["ICULOS"], errors="raise").to_numpy(dtype=float)
        if not len(original_y) or not np.isin(original_y, (0, 1)).all():
            raise ValueError(f"Bad original label series: {patient_id}")
        if not np.isfinite(original_t).all() or np.any(np.diff(original_t) <= 0):
            raise ValueError(f"Non-monotonic ICU times: {patient_id}")
        total_hours += len(original_y)
        if len(original_y) < 24:
            short += 1
            rows = 0
        else:
            scoreable += 1
            rows = len(original_y) - 23
        sl = slice(cursor, cursor + rows)
        if rows and not (
            np.array_equal(ids[sl], np.repeat(patient_id, rows))
            and np.array_equal(labels[sl], original_y[23:])
            and np.array_equal(hours[sl], original_t[23:])
        ):
            raise ValueError(f"Patient, time, or label misalignment: {patient_id}")
        alarm, silent, ideal = utility_components(original_y)
        best_sum += float(np.where(ideal == 1, alarm, silent).sum())
        inactive_sum += float(silent.sum())
        for name, all_scores in scores.items():
            local = all_scores[sl]
            for cutoff in THRESHOLDS:
                pred = np.zeros(len(original_y), dtype=bool)
                if rows:
                    pred[23:] = local >= cutoff
                utility_sums[name, cutoff] += float(
                    np.where(pred, alarm, silent).sum())
        cursor += rows
        if count % 1000 == 0:
            print(f"Utility processed {count:,}/{len(val_ids):,} Validation patients", flush=True)
    if cursor != n:
        raise ValueError("Archived score rows were not consumed exactly once")
    denominator = best_sum - inactive_sum
    if denominator <= 0:
        raise ValueError("Official normalization denominator nonpositive")
    rows = [
        {
            "model": model, "score_threshold": threshold,
            "normalized_utility": (total - inactive_sum) / denominator,
            "unnormalized_utility": total,
        }
        for (model, threshold), total in utility_sums.items()
    ]
    table = pd.DataFrame(rows).sort_values(["model", "score_threshold"])
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_OUTPUT, index=False)
    report = {
        "scope": "original_locked_validation_all_patient_hours",
        "original_validation_patients": len(val_ids),
        "patients_with_24hour_windows": scoreable,
        "patients_fewer_than_24_rows": short,
        "total_patient_hours": total_hours,
        "scoreable_rows": n,
        "test_patients_evaluated": 0,
        "first_23_hours": "predict zero because insufficient 24-row observations",
        "thresholds": THRESHOLDS,
        "original_sepsis_label_shifted_again": False,
        "official_best_utility": best_sum,
        "official_inaction_utility": inactive_sum,
        "normalization_denominator": denominator,
        "results": rows,
        "interpretation": (
            "Uses the existing PhysioNet 2019 official-style utility formula and "
            "full original Validation timelines. This is NOT a Challenge submission. "
            "Prespecified score cutoff grid is diagnostic: choosing a cutoff from "
            "the same repeatedly used Validation induces selection bias. "
            "Scores are not calibrated clinical probabilities. No 6-hour alert "
            "cooldown is applied to official utility. Test stays held out."),
        "reference": "https://github.com/physionetchallenges/evaluation-2019",
    }
    JSON_OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("At diagnostic cutoff 0.2:")
    print(table.loc[np.isclose(table["score_threshold"], 0.2)].to_string(index=False),
          flush=True)
    print("Saved:", CSV_OUTPUT, JSON_OUTPUT, "; Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
