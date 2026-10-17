"""Validate frozen Train-only LSTM preprocessing on Validation and Test patients.

Reads (never refits) models/lstm_preprocessing_stats.json.
No model training, test scoring, or edits to input data.
Run from project root: python src/preprocessing/validate_lstm_splits.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_sequences import (
    DEFAULT_MANIFEST, DEFAULT_ROOT, DEFAULT_STATS, FEATURES, WINDOW,
    make_sequences, patient_path, read_manifest,
)


def validate(stats: dict, groups: dict, root: Path, limit: int | None = None) -> pd.DataFrame:
    expected_train = len(groups["train"])
    if stats.get("train_patients_used") != expected_train:
        raise ValueError("Statistics not fitted on FULL manifest Train split")
    if stats.get("train_patients_in_manifest") != expected_train:
        raise ValueError("Statistics were generated for a different Train split size")
    if stats.get("features") != FEATURES or stats.get("window_rows") != WINDOW:
        raise ValueError("Preprocessing schema mismatch")
    rows = []
    for split in ("validation", "test"):
        ids = groups[split] if limit is None else groups[split][:limit]
        patient_count = 0
        short_count = 0
        samples = 0
        positives = 0
        for i, patient_id in enumerate(ids, 1):
            raw = pd.read_csv(patient_path(root, patient_id), sep="|")
            ordered = raw.sort_values("ICULOS", kind="stable").reset_index(drop=True)
            x, y = make_sequences(raw, stats)
            expected_count = max(len(raw) - WINDOW + 1, 0)
            if x.shape != (expected_count, WINDOW, len(FEATURES) * 4):
                raise AssertionError(f"Wrong X shape: {split}/{patient_id}: {x.shape}")
            if y.shape != (expected_count,):
                raise AssertionError(f"Wrong y shape: {split}/{patient_id}: {y.shape}")
            if not np.isfinite(x).all():
                raise AssertionError(f"Nonfinite LSTM values: {split}/{patient_id}")
            expected_y = ordered["SepsisLabel"].to_numpy(dtype=int)[WINDOW - 1:]
            if not np.array_equal(y, expected_y):
                raise AssertionError(f"Labels changed: {split}/{patient_id}")
            masks = x[:, -1, 1::4]  # last-timepoint original-observed indicators
            if masks.size and not np.isin(masks, [0.0, 1.0]).all():
                raise AssertionError(f"Nonbinary masks: {split}/{patient_id}")
            patient_count += 1
            short_count += int(len(raw) < WINDOW)
            samples += len(y)
            positives += int(y.sum())
            if i % 1000 == 0:
                print(f"{split}: checked {i:,}/{len(ids):,}", flush=True)
        rows.append({
            "split": split,
            "patients_checked": patient_count,
            "patients_under_24_rows": short_count,
            "sequences_24x64": samples,
            "positive_sequences": positives,
            "positive_sequence_rate_pct": round(100 * positives / samples, 3) if samples else np.nan,
            "finite_inputs": True,
            "target_matches_raw": True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--train-stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--max-patients-per-split", type=int, default=None,
                        help="Limit validation and test patients for a smoke test")
    parser.add_argument("--output", type=Path, default=Path("reports/lstm_split_validation.csv"))
    args = parser.parse_args()
    groups = read_manifest(args.split_manifest)
    if not args.train_stats.is_file():
        raise FileNotFoundError(args.train_stats)
    stats = json.loads(args.train_stats.read_text(encoding="utf-8"))
    print("Checking frozen Train-only medians / IQR on Validation and Test", flush=True)
    result = validate(stats, groups, args.data_root, args.max_patients_per_split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved report: {args.output}")
    print("PASS: frozen Train stats, 24x64 finite inputs, patient boundaries, original targets")
    print("Note: this verifies data transformation, not predictive performance.")


if __name__ == "__main__":
    main()
