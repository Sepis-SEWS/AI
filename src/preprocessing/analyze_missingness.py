"""Analyze missingness persistence and 24-row observation windows in PhysioNet 2019.

Run from repository root:
    python src/preprocessing/analyze_missingness.py

This script only reads PSV files. It does not impute values, shift labels,
change split manifests, or include future measurements in window statistics.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total",
    "Platelets", "FiO2", "BUN", "Glucose", "pH",
]
ROOT = Path("data/data-sepsis/training")
REPORT = Path("reports/missingness_window24.csv")


def analyze_patient(frame: pd.DataFrame) -> dict[str, dict]:
    """Return observed times, internal gap lengths, and 24-row window counts."""
    needed = set(FEATURES) | {"ICULOS"}
    absent = needed - set(frame.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    data = frame.sort_values("ICULOS", kind="stable")
    times = pd.to_numeric(data["ICULOS"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Invalid/duplicate ICULOS")
    result = {}
    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce").to_numpy(dtype=float)
        observed = np.isfinite(values).astype(np.int8)
        observed_times = times[observed == 1]
        # Number of hourly rows between consecutive actual measurements.
        observed_positions = np.flatnonzero(observed)
        gaps = (np.diff(observed_positions) - 1).tolist()
        # Match XGBoost: full 24-row windows, ending at rows 23..N-1.
        window_counts = (
            np.convolve(observed, np.ones(24, dtype=np.int32), mode="valid")
            if len(observed) >= 24 else np.empty(0, dtype=np.int32)
        )
        result[feature] = {
            "observations": int(observed.sum()),
            "hours": int(len(observed)),
            "ever_observed": int(observed.sum() > 0),
            "gaps": gaps,
            "window_counts": window_counts,
            "observed_times": observed_times,
        }
    return result


def summarize(paths: list[Path], source: str) -> list[dict]:
    totals = defaultdict(lambda: {
        "patients": 0, "patients_observed": 0, "hours": 0, "observations": 0,
        "windows": 0, "zero": 0, "one": 0, "two": 0, "three_plus": 0,
        "gaps": [], "window_observations": 0,
    })
    for i, path in enumerate(paths, start=1):
        frame = pd.read_csv(path, sep="|")
        patient_stats = analyze_patient(frame)
        for feature, info in patient_stats.items():
            item = totals[feature]
            item["patients"] += 1
            item["patients_observed"] += info["ever_observed"]
            item["hours"] += info["hours"]
            item["observations"] += info["observations"]
            counts = info["window_counts"]
            item["windows"] += len(counts)
            item["zero"] += int(np.count_nonzero(counts == 0))
            item["one"] += int(np.count_nonzero(counts == 1))
            item["two"] += int(np.count_nonzero(counts == 2))
            item["three_plus"] += int(np.count_nonzero(counts >= 3))
            item["window_observations"] += int(counts.sum())
            item["gaps"].extend(info["gaps"])
        if i % 1000 == 0:
            print(f"{source}: read {i:,}/{len(paths):,} patients", flush=True)
    rows = []
    for feature in FEATURES:
        item = totals[feature]
        gaps = np.asarray(item["gaps"], dtype=float)
        windows = item["windows"]
        rows.append({
            "source": source,
            "feature": feature,
            "patients": item["patients"],
            "patients_ever_observed_pct": round(100 * item["patients_observed"] / item["patients"], 2),
            "pooled_missing_pct": round(100 * (1 - item["observations"] / item["hours"]), 2),
            "mean_measurements_per_patient": round(item["observations"] / item["patients"], 2),
            "full_24h_windows": windows,
            "window_zero_observed_pct": round(100 * item["zero"] / windows, 2) if windows else np.nan,
            "window_one_observed_pct": round(100 * item["one"] / windows, 2) if windows else np.nan,
            "window_two_observed_pct": round(100 * item["two"] / windows, 2) if windows else np.nan,
            "window_three_plus_pct": round(100 * item["three_plus"] / windows, 2) if windows else np.nan,
            "mean_observations_per_24h_window": round(item["window_observations"] / windows, 2) if windows else np.nan,
            "internal_gap_p50_rows": float(np.percentile(gaps, 50)) if gaps.size else np.nan,
            "internal_gap_p95_rows": float(np.percentile(gaps, 95)) if gaps.size else np.nan,
            "internal_gap_max_rows": int(np.max(gaps)) if gaps.size else np.nan,
            "internal_gap_count": int(gaps.size),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=REPORT)
    parser.add_argument("--max-patients-per-source", type=int, default=None,
                        help="Optional smoke test only; default processes all patients")
    args = parser.parse_args()
    all_rows = []
    for source in ("training_setA", "training_setB"):
        paths = sorted((args.data_root / source).glob("*.psv"))
        if args.max_patients_per_source is not None:
            paths = paths[:args.max_patients_per_source]
        if not paths:
            raise FileNotFoundError(f"No PSV files in {args.data_root / source}")
        print(f"Analyzing {source}: {len(paths):,} patients", flush=True)
        all_rows.extend(summarize(paths, source))
    output = pd.DataFrame(all_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(output[[
        "source", "feature", "pooled_missing_pct",
        "window_zero_observed_pct", "mean_observations_per_24h_window",
        "internal_gap_p95_rows",
    ]].to_string(index=False))
    print("Note: internal gap is intervening ROWS between real measurements;")
    print("initial/trailing missing spans are reflected by window-zero rates.")
    print("No forward-fill thresholds have been selected.")


if __name__ == "__main__":
    main()
