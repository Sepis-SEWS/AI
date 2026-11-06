"""Prepare leak-safe LSTM Group 5-Fold on original locked TRAIN patients only.

Usage:
  python src/training/prepare_lstm_group_cv.py
  python src/training/prepare_lstm_group_cv.py --fit-stats-fold 1

Creates 5 fold manifests in data/splits/lstm_cv/ without changing the
original manifest. Fold validation is drawn ONLY from original Train.
Original Validation and Test remain outside all fold fits/scores.

Each fold REQUIRES separately fitted preprocessing medians/IQR on that fold's
training patients. --fit-stats-fold 1 performs the expensive fold-1 fit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))
from lstm_sequences import fit_train_stats, read_manifest  # noqa: E402

MANIFEST = Path("data/splits/patient_splits.json")
INDEX = Path("data/splits/patient_index.csv")
ROOT = Path("data/data-sepsis/training")
FOLD_DIR = Path("data/splits/lstm_cv")
STATS_DIR = Path("models")
SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-stats-fold", type=int, default=None,
                        help="1..5: fit fold-specific statistics from only its train patients")
    args = parser.parse_args()
    if args.fit_stats_fold is not None and args.fit_stats_fold not in range(1, 6):
        parser.error("--fit-stats-fold must be 1..5")
    groups = read_manifest(MANIFEST)
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate ID in patient index")
    index = index.set_index("patient_id")
    train_ids = groups["train"]
    if len(train_ids) != 25814 or len(groups["validation"]) != 6454 or len(groups["test"]) != 8068:
        raise ValueError("Locked original cohort count mismatch")
    if not set(train_ids).issubset(set(index.index)):
        raise ValueError("Missing Train IDs in index")
    train_index = index.loc[train_ids]
    strata = (train_index["source"].astype(str) + "/" +
              train_index["positive"].astype(int).astype(str)).to_numpy()
    fold_maker = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_sets = []
    original_val = set(groups["validation"])
    original_test = set(groups["test"])
    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    for fold, (fit_indices, held_indices) in enumerate(
            fold_maker.split(np.zeros(len(train_ids)), strata), start=1):
        fit_ids = [train_ids[i] for i in fit_indices]
        held_ids = [train_ids[i] for i in held_indices]
        if (set(fit_ids) & set(held_ids) or
                (set(fit_ids) | set(held_ids)) != set(train_ids)):
            raise ValueError("Fold does not partition original Train")
        if (set(fit_ids) | set(held_ids)) & (original_val | original_test):
            raise ValueError("Original Validation or Test leaked into CV")
        fold_sets.append(set(held_ids))
        target = FOLD_DIR / f"fold{fold}.json"
        # Train script's 'test' field remains the original locked Test, but
        # neither the stats fitter nor trainer opens its PSV files.
        payload = {
            "split": {"train": fit_ids, "validation": held_ids,
                      "test": groups["test"]},
            "cv_metadata": {
                "fold": fold, "seed": SEED,
                "fold_validation_source": "original locked Train only",
                "original_validation_held_out": len(groups["validation"]),
                "test_never_evaluated": True,
            },
        }
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != payload:
                raise FileExistsError(f"Existing fold manifest differs: {target}")
        else:
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"Fold {fold}: train={len(fit_ids):,}, "
              f"fold_validation={len(held_ids):,}; manifest={target}", flush=True)
    if set.union(*fold_sets) != set(train_ids) or (
            sum(len(s) for s in fold_sets) != len(train_ids)):
        raise ValueError("Each original Train patient must be in exactly one held-out fold")
    if args.fit_stats_fold is not None:
        number = args.fit_stats_fold
        fold_data = read_manifest(FOLD_DIR / f"fold{number}.json")
        target = STATS_DIR / f"lstm_cv_fold{number}_stats.json"
        if target.exists():
            prior = json.loads(target.read_text(encoding="utf-8"))
            if (prior.get("train_patients_used") != len(fold_data["train"]) or
                    prior.get("train_patients_in_manifest") != len(fold_data["train"])):
                raise ValueError(f"Existing fold stats incompatible: {target}")
            print(f"Fold {number} stats already exist: {target}", flush=True)
        else:
            stats = fit_train_stats(fold_data["train"], ROOT)
            stats["train_patients_used"] = len(fold_data["train"])
            stats["train_patients_in_manifest"] = len(fold_data["train"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                              encoding="utf-8")
            print(f"Saved fold {number} Train-only stats: {target}", flush=True)
    print("CV manifests ready. Original Validation and Test NOT accessed.")


if __name__ == "__main__":
    main()
