"""Train-only LSTM value calibration and causal 24-row sequence extraction.

Run: python src/preprocessing/lstm_sequences.py
Input split manifest must contain 'split' and nonempty patient IDs.
Raw patient files and split manifest are never altered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import FEATURES, preprocess_patient

DEFAULT_MANIFEST = Path("data/splits/patient_splits.json")
DEFAULT_ROOT = Path("data/data-sepsis/training")
DEFAULT_STATS = Path("models/lstm_preprocessing_stats.json")
WINDOW = 24


def read_manifest(path: Path) -> dict[str, list[str]]:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Split manifest missing or empty: {path}. Restore local full split first.")
    source = json.loads(path.read_text(encoding="utf-8"))
    groups = source.get("split", {})
    if not all(isinstance(groups.get(k), list) and groups[k]
               for k in ("train", "validation", "test")):
        raise ValueError("Manifest requires nonempty split.train/validation/test lists")
    sets = [set(groups[k]) for k in ("train", "validation", "test")]
    if any(len(group) != len(ids) for group, ids in zip(sets, (groups[k] for k in ("train", "validation", "test")))):
        raise ValueError("Duplicate IDs inside split")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Patient appears in multiple splits")
    return groups


def patient_path(root: Path, patient_id: str) -> Path:
    parts = patient_id.split("/")
    if len(parts) != 2 or parts[0] not in ("training_setA", "training_setB"):
        raise ValueError(f"Unexpected patient ID: {patient_id}")
    file = root / parts[0] / f"{parts[1]}.psv"
    if not file.is_file():
        raise FileNotFoundError(file)
    return file


def fit_train_stats(patient_ids: list[str], root: Path) -> dict:
    """Fit robust medians and IQR scales from RAW finite TRAIN observations only."""
    values = {f: [] for f in FEATURES}
    for index, patient_id in enumerate(patient_ids, 1):
        df = pd.read_csv(patient_path(root, patient_id), sep="|")
        for name in FEATURES:
            observed = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
            observed = observed[np.isfinite(observed)]
            if observed.size:
                values[name].append(observed)
        if index % 1000 == 0:
            print(f"Training statistics: {index:,}/{len(patient_ids):,}", flush=True)
    medians, scales, counts = {}, {}, {}
    for name in FEATURES:
        obs = np.concatenate(values[name]) if values[name] else np.empty(0)
        if not len(obs):
            raise ValueError(f"No finite observations for train variable {name}")
        median = float(np.median(obs))
        iqr = float(np.percentile(obs, 75) - np.percentile(obs, 25))
        medians[name] = median
        scales[name] = iqr if np.isfinite(iqr) and iqr > 1e-12 else 1.0
        counts[name] = int(len(obs))
    return {
        "window_rows": WINDOW, "features": FEATURES,
        "median": medians, "scale_iqr": scales, "train_observation_counts": counts,
        "age_cap_hours": 48.0,
        "channels_per_feature": ["scaled_value", "original_mask", "age_hours_scaled", "never_observed"],
        "fit": "finite raw measurements of train patients only",
    }


def make_sequences(frame: pd.DataFrame, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,24,64) float32 features and end-of-window SepsisLabel.

    Uses completed 24-row windows, as in the existing XGBoost baseline.
    age NaN (never measured) is encoded with a distinct binary channel.
    """
    if stats["features"] != FEATURES or stats["window_rows"] != WINDOW:
        raise ValueError("Feature schema/window mismatch")
    clean = preprocess_patient(frame)
    features = []
    for name in FEATURES:
        median = float(stats["median"][name])
        scale = float(stats["scale_iqr"][name])
        if not np.isfinite(median) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid training statistics for {name}")
        val = clean[name].to_numpy(float)
        val = np.where(np.isfinite(val), val, median)
        scaled = (val - median) / scale
        mask = clean[f"{name}_mask"].to_numpy(float)
        age = clean[f"{name}_age_hours"].to_numpy(float)
        never = ~np.isfinite(age)
        # Only past-observed age, clipped to control extreme ages.
        age = np.clip(np.where(never, 0.0, age), 0.0, float(stats["age_cap_hours"])) / float(stats["age_cap_hours"])
        features.extend((scaled, mask, age, never.astype(float)))
    # time-axis first; group all 4 channels per variable
    matrix = np.stack(features, axis=1).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite LSTM input after preprocessing")
    label = clean["SepsisLabel"].to_numpy(int)
    if not np.isin(label, [0, 1]).all():
        raise ValueError("SepsisLabel not binary")
    if len(matrix) < WINDOW:
        return np.empty((0, WINDOW, len(FEATURES) * 4), dtype=np.float32), np.empty(0, dtype=np.int8)
    x = np.stack([matrix[i - WINDOW + 1:i + 1] for i in range(WINDOW - 1, len(matrix))])
    y = label[WINDOW - 1:].astype(np.int8)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stats-output", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--max-train-patients", type=int, default=None,
                        help="Smoke test only; statistics MUST be refitted on full train before model evaluation")
    args = parser.parse_args()
    groups = read_manifest(args.split_manifest)
    ids = groups["train"]
    if args.max_train_patients is not None:
        ids = ids[:args.max_train_patients]
    if not ids:
        raise ValueError("No training patients")
    print(f"Train-only statistics: {len(ids):,} patients", flush=True)
    stats = fit_train_stats(ids, args.data_root)
    # Include explicit scope, to forbid treating smoke-test statistics as full.
    stats["train_patients_used"] = len(ids)
    stats["train_patients_in_manifest"] = len(groups["train"])
    target = args.stats_output
    if args.max_train_patients is not None:
        target = target.with_name(target.stem + "_sample" + str(len(ids)) + target.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    first = pd.read_csv(patient_path(args.data_root, ids[0]), sep="|")
    x, y = make_sequences(first, stats)
    print(f"Saved train-only stats: {target}")
    print(f"First train patient: X shape={x.shape}, y shape={y.shape}, finite={np.isfinite(x).all()}")
    print("No validation/test observations used for median or scaling.")


if __name__ == "__main__":
    main()
