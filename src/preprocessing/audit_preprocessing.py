"""SEWS AI-3: audit leading/trailing missingness and numeric value distributions.

Read-only analysis of all patient PSV files, split by source. No imputation
or automatic outlier removal. Run from repository root:
    python src/preprocessing/audit_preprocessing.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [
    "HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total",
    "Platelets", "FiO2", "BUN", "Glucose", "pH",
]
ROOT = Path("data/data-sepsis/training")


def audit_patient(frame: pd.DataFrame) -> dict[str, dict]:
    """Summarize one patient; row gaps and elapsed ICULOS times are separate."""
    missing = set(FEATURES + ["ICULOS"]) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = frame.sort_values("ICULOS", kind="stable").reset_index(drop=True)
    times = pd.to_numeric(data["ICULOS"], errors="coerce").to_numpy(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("ICULOS must be finite and strictly increasing")
    result = {}
    for feature in FEATURES:
        raw = data[feature]
        values = pd.to_numeric(raw, errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        indices = np.flatnonzero(finite)
        observed = values[finite]
        never = len(indices) == 0
        initial_rows = int(indices[0]) if not never else len(values)
        trailing_rows = int(len(values) - 1 - indices[-1]) if not never else len(values)
        # Maximum age of a measured value reached before next measurement/end.
        if len(indices):
            elapsed = np.diff(times[indices]) if len(indices) > 1 else np.empty(0)
            stale_at_end = float(times[-1] - times[indices[-1]])
        else:
            elapsed = np.empty(0)
            stale_at_end = np.nan
        result[feature] = {
            "never": int(never),
            "initial_rows": initial_rows,
            "trailing_rows": trailing_rows,
            "trailing_elapsed": stale_at_end,
            "between_elapsed": elapsed,
            "values": observed,
            "negative": int(np.count_nonzero(observed < 0)),
            "inf": int(np.count_nonzero(np.isinf(values))),
            "invalid_nonnull": int(np.count_nonzero(raw.notna().to_numpy() & ~finite)),
        }
    return result


def q(values: list[float], percentile: float) -> float:
    return round(float(np.percentile(values, percentile)), 2) if values else np.nan


def analyze(root: Path, max_patients: int | None) -> pd.DataFrame:
    rows = []
    for source in ("training_setA", "training_setB"):
        files = sorted((root / source).glob("*.psv"))
        if max_patients is not None:
            files = files[:max_patients]
        if not files:
            raise FileNotFoundError(f"No PSV files found in {root / source}")
        totals = {f: {
            "patients": 0, "never": 0, "negative": 0, "inf": 0,
            "invalid_nonnull": 0, "initial": [], "trailing": [],
            "trailing_elapsed": [], "between_elapsed": [], "observed": [],
        } for f in FEATURES}
        print(f"{source}: {len(files):,} patients", flush=True)
        for i, path in enumerate(files, 1):
            stats = audit_patient(pd.read_csv(path, sep="|"))
            for feature in FEATURES:
                record = totals[feature]
                item = stats[feature]
                record["patients"] += 1
                record["never"] += item["never"]
                record["negative"] += item["negative"]
                record["inf"] += item["inf"]
                record["invalid_nonnull"] += item["invalid_nonnull"]
                if not item["never"]:
                    record["initial"].append(item["initial_rows"])
                    record["trailing"].append(item["trailing_rows"])
                    record["trailing_elapsed"].append(item["trailing_elapsed"])
                    record["between_elapsed"].extend(item["between_elapsed"].tolist())
                    record["observed"].extend(item["values"].tolist())
            if i % 1000 == 0:
                print(f"  read {i:,}/{len(files):,}", flush=True)
        for feature in FEATURES:
            t = totals[feature]
            v = t["observed"]
            rows.append({
                "source": source, "feature": feature, "patients": t["patients"],
                "never_observed_pct": round(100*t["never"]/t["patients"], 2),
                "leading_missing_rows_p50": q(t["initial"], 50),
                "leading_missing_rows_p95": q(t["initial"], 95),
                "trailing_missing_rows_p50": q(t["trailing"], 50),
                "trailing_missing_rows_p95": q(t["trailing"], 95),
                "last_measurement_age_hours_p95": q(t["trailing_elapsed"], 95),
                "measurement_interval_hours_p50": q(t["between_elapsed"], 50),
                "measurement_interval_hours_p95": q(t["between_elapsed"], 95),
                "observed_count": len(v),
                "observed_p01": q(v, 1), "observed_p50": q(v, 50),
                "observed_p99": q(v, 99),
                "observed_min": min(v) if v else np.nan,
                "observed_max": max(v) if v else np.nan,
                "negative_values": t["negative"],
                "infinite_values": t["inf"],
                "invalid_nonnull_values": t["invalid_nonnull"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("reports/preprocessing_audit.csv"))
    parser.add_argument("--max-patients-per-source", type=int, default=None)
    args = parser.parse_args()
    report = analyze(args.data_root, args.max_patients_per_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")
    print(report[[
        "source", "feature", "never_observed_pct",
        "leading_missing_rows_p95", "trailing_missing_rows_p95",
        "measurement_interval_hours_p95", "observed_p01", "observed_p99",
        "negative_values",
    ]].to_string(index=False))
    print("Descriptive audit only: no values have been changed or discarded.")


if __name__ == "__main__":
    main()
