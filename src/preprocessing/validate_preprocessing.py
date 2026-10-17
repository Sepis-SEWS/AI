"""Audit the age-limited causal preprocessing on real PhysioNet files.

Run from project root:
  python src/preprocessing/validate_preprocessing.py
  python src/preprocessing/validate_preprocessing.py --max-patients-per-source 100

Read-only: does not modify raw PSV data, labels, or split manifests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import FEATURES, FFILL_MAX_AGE_HOURS, preprocess_patient

ROOT = Path("data/data-sepsis/training")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--max-patients-per-source", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("reports/preprocessing_validation.csv"))
    args = parser.parse_args()
    rows = []
    for source in ("training_setA", "training_setB"):
        files = sorted((args.data_root / source).glob("*.psv"))
        if args.max_patients_per_source is not None:
            files = files[:args.max_patients_per_source]
        if not files:
            raise FileNotFoundError(f"No PSV files: {args.data_root / source}")
        counts = {name: {"observed": 0, "filled": 0, "unfilled": 0, "records": 0} for name in FEATURES}
        print(f"{source}: {len(files):,} patients", flush=True)
        for i, path in enumerate(files, 1):
            original = pd.read_csv(path, sep="|")
            processed = preprocess_patient(original)
            ordered = original.sort_values("ICULOS", kind="stable").reset_index(drop=True)
            if not processed["SepsisLabel"].equals(ordered["SepsisLabel"]):
                raise AssertionError(f"SepsisLabel changed: {path}")
            for name in FEATURES:
                raw = processed[f"{name}_raw"]
                result = processed[name]
                mask = processed[f"{name}_mask"].astype(bool)
                age = processed[f"{name}_age_hours"]
                expected_observed = raw.notna()
                if not mask.equals(expected_observed):
                    raise AssertionError(f"Mask mismatch: {path} {name}")
                if not result[mask].equals(raw[mask]):
                    raise AssertionError(f"Original observation changed: {path} {name}")
                limit = FFILL_MAX_AGE_HOURS[name]
                filled = ~mask & result.notna()
                if limit is None and filled.any():
                    raise AssertionError(f"Sparse variable filled unexpectedly: {path} {name}")
                if limit is not None and (age[filled] > limit).any():
                    raise AssertionError(f"Expired value used: {path} {name}")
                tally = counts[name]
                tally["records"] += len(raw)
                tally["observed"] += int(mask.sum())
                tally["filled"] += int(filled.sum())
                tally["unfilled"] += int(result.isna().sum())
            if i % 1000 == 0:
                print(f"  checked {i:,}/{len(files):,}", flush=True)
        for name in FEATURES:
            t = counts[name]
            rows.append({
                "source": source,
                "feature": name,
                "ffill_limit_hours": FFILL_MAX_AGE_HOURS[name],
                "records": t["records"],
                "observed": t["observed"],
                "filled": t["filled"],
                "remaining_missing": t["unfilled"],
                "remaining_missing_pct": round(100 * t["unfilled"] / t["records"], 2),
            })
    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(report[["source", "feature", "filled", "remaining_missing_pct"]].to_string(index=False))
    print("PASS: original observations, labels, masks, and age limits preserved")


if __name__ == "__main__":
    main()
