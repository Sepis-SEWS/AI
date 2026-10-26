"""SEWS AI-4 XGBoost baseline using the LOCKED patient split.

Smoke test (source/label-stratified within the original splits):
 python src/training/train_xgboost_manifest.py --max-train 300 --max-validation 100
Full training (may require significant RAM/time):
 python src/training/train_xgboost_manifest.py

Never loads/evaluates Test patients. Produces a separate model and report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from train_xgboost import FEATURES, VARS, load_group


def subset(ids: list[str], index: pd.DataFrame, limit: int | None, seed: int) -> list[str]:
    """Keep original manifest membership, sample proportionally by source/label."""
    if limit is None or limit >= len(ids):
        return ids
    if limit <= 0:
        raise ValueError("Sample size must be >0")
    table = index.set_index("patient_id").loc[ids]
    groups = (table["source"].astype(str) + "/" + table["positive"].astype(int).astype(str))
    # Deterministic, approximately stratified sample without requiring a minimum class count.
    rng = np.random.default_rng(seed)
    counts = groups.value_counts().sort_index()
    allocations = counts * (limit / len(ids))
    taken = np.floor(allocations).astype(int)
    left = limit - int(taken.sum())
    for key in (allocations - taken).sort_values(ascending=False, kind="stable").index[:left]:
        taken.loc[key] += 1
    chosen: list[str] = []
    for group, number in taken.items():
        choices = groups.index[groups == group].to_numpy()
        chosen.extend(rng.choice(choices, size=int(number), replace=False).tolist())
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("data/splits/patient_splits.json"))
    parser.add_argument("--patient-index", type=Path,
                        default=Path("data/splits/patient_index.csv"))
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.split_manifest.is_file() or args.split_manifest.stat().st_size == 0:
        raise ValueError("Patient split manifest missing or empty; don't regenerate or overwrite it")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    groups = manifest["split"]
    ids_train, ids_val, ids_test = (groups[k] for k in ("train", "validation", "test"))
    if len(set(ids_train + ids_val + ids_test)) != sum(len(x) for x in (ids_train, ids_val, ids_test)):
        raise ValueError("Patient IDs overlap across splits")
    index = pd.read_csv(args.patient_index)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient ID in patient index")
    train_ids = subset(ids_train, index, args.max_train, args.seed)
    val_ids = subset(ids_val, index, args.max_validation, args.seed + 1)
    files = index.set_index("patient_id")["path"].to_dict()
    train_paths = [Path(files[pid]) for pid in train_ids]
    val_paths = [Path(files[pid]) for pid in val_ids]
    print(f"Locked manifest: train {len(train_ids):,}/{len(ids_train):,}, "
          f"validation {len(val_ids):,}/{len(ids_val):,}; test HELD OUT", flush=True)
    print("Generating 160 raw-observation features ...", flush=True)
    x_train, y_train = load_group(train_paths)
    x_val, y_val = load_group(val_paths)
    if len(y_train) == 0 or len(np.unique(y_train)) != 2:
        raise ValueError("Training samples require both classes; increase --max-train")
    if len(y_val) == 0 or len(np.unique(y_val)) != 2:
        raise ValueError("Validation samples require both classes; increase --max-validation")
    weight = float((len(y_train) - int(y_train.sum())) / int(y_train.sum()))
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=weight, eval_metric="logloss",
        n_jobs=-1, random_state=args.seed,
    )
    print(f"Training: {len(y_train):,} samples, positives {int(y_train.sum()):,}", flush=True)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_val)[:, 1]
    report = {
        "experiment": "xgboost_locked_patient_split",
        "full_train": len(train_ids) == len(ids_train),
        "full_validation": len(val_ids) == len(ids_val),
        "train_patients": len(train_ids),
        "validation_patients": len(val_ids),
        "test_patients_evaluated": 0,
        "train_samples": len(y_train),
        "validation_samples": len(y_val),
        "validation_positive_rate": float(np.mean(y_val)),
        "validation_auroc": float(roc_auc_score(y_val, prob)),
        "validation_ap": float(average_precision_score(y_val, prob)),
        "features": len(FEATURES),
        "seed": args.seed,
    }
    scope = "full" if report["full_train"] and report["full_validation"] else (
        f"sample_train{len(train_ids)}_val{len(val_ids)}"
    )
    Path("reports").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    output = Path("reports") / f"xgboost_manifest_{scope}.json"
    artifact = Path("models") / f"xgboost_manifest_{scope}.joblib"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"model": model, "features": FEATURES, "selected_columns": VARS,
                 "window_rows": 24, "experiment": report}, artifact)
    print(f"Validation AUROC={report['validation_auroc']:.4f}, "
          f"AP={report['validation_ap']:.4f}, "
          f"positive_rate={report['validation_positive_rate']:.4f}")
    print(f"Saved report: {output}")
    print(f"Saved model: {artifact}")
    print("No Test labels or features were accessed. No calibration or final scoring.")


if __name__ == "__main__":
    main()
