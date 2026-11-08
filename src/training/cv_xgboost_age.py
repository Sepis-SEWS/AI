"""SEWS AI-4 patient-grouped 5-fold CV on LOCKED TRAIN patients only.

Compare the original 160 features with 160 + 16 causal age_hours.
Existing feature caches and missingness checkpoints are reused.
Each fitted fold is checkpointed independently (safe to resume).

Smoke:
  python src/training/cv_xgboost_age.py --max-train 500 --folds 2
Full:
  python src/training/cv_xgboost_age.py

Validation and Test patients are never read. Do not use fold results as
independent confirmation after selecting age on the prior Validation set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, fingerprint, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS
from train_xgboost_manifest import subset

PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
    eval_metric="logloss", n_jobs=-1,
)
OUT_DIR = Path("reports/xgboost_cv_age")
AGE_INDICES = [EXTRA_NAMES.index(f"{v}_age_hours") for v in VARS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    if not MANIFEST.is_file() or MANIFEST.stat().st_size == 0:
        raise FileNotFoundError("Locked manifest missing. Never recreate it for CV.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    train_all, validation_all, test_all = (manifest[k] for k in ("train", "validation", "test"))
    if len(set(train_all + validation_all + test_all)) != sum(map(
            len, (train_all, validation_all, test_all))):
        raise ValueError("Overlapping manifest cohorts")
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs")
    train_ids = subset(train_all, index, args.max_train, args.seed)
    if set(train_ids) & (set(validation_all) | set(test_all)):
        raise AssertionError("Forbidden holdout patients in CV")
    rows = index.set_index("patient_id").loc[train_ids]
    paths = [Path(p) for p in rows["path"]]
    patient_strata = (rows["source"].astype(str) + "/" +
                      rows["positive"].astype(int).astype(str)).to_numpy()
    counts = pd.Series(patient_strata).value_counts()
    if counts.min() < args.folds:
        raise ValueError(f"Need >= {args.folds} patients per source/label stratum; "
                         f"got {counts.to_dict()}")

    print(f"TRAIN-only CV: {len(train_ids):,} / {len(train_all):,} patients; "
          f"{args.folds} folds. Validation and Test HELD OUT.", flush=True)
    base, y, patient_groups = load_or_build("train", train_ids, paths, CACHE_DIR, False)
    extras = chunks_for("train", train_ids, paths, CACHE_DIR, False)
    if base.shape[0] != extras.shape[0] or base.shape[0] != len(patient_groups):
        raise ValueError("Unaligned cache arrays")
    # Cached rows are written in exactly the manifest-selected patient order.
    observed = list(dict.fromkeys(patient_groups.tolist()))
    expected = [pid for pid in train_ids if pid in set(observed)]
    if observed != expected:
        raise ValueError("Feature rows do not align to selected manifest patient IDs")
    if not set(patient_groups).issubset(set(train_ids)):
        raise ValueError("Unexpected patient in feature cache")
    if not np.isfinite(extras).all():
        raise ValueError("Missingness metadata must be finite")

    key = hashlib.sha256()
    key.update(Path(__file__).read_bytes())
    key.update(Path(__file__).with_name("experiment_missingness.py").read_bytes())
    key.update(fingerprint(paths, train_ids).encode())
    key.update(json.dumps({"strata": patient_strata.tolist(), "folds": args.folds,
                           "seed": args.seed, "params": PARAMS}, sort_keys=True).encode())
    run_id = key.hexdigest()[:20]
    work = OUT_DIR / f"train{len(train_ids)}_k{args.folds}_{run_id}"
    work.mkdir(parents=True, exist_ok=True)
    variants = {
        "baseline_160": (FEATURES, base),
        "age_176": (FEATURES + [f"{v}_age_hours" for v in VARS],
                    np.column_stack((base, extras[:, AGE_INDICES]))),
    }
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    completed = []
    for fold_num, (patient_train_idx, patient_val_idx) in enumerate(
            folds.split(np.arange(len(train_ids)), patient_strata), start=1):
        fold_train_ids = set(np.asarray(train_ids)[patient_train_idx])
        fold_val_ids = set(np.asarray(train_ids)[patient_val_idx])
        if fold_train_ids & fold_val_ids:
            raise AssertionError("Patient overlap in fold")
        train_mask = np.isin(patient_groups, list(fold_train_ids))
        val_mask = np.isin(patient_groups, list(fold_val_ids))
        if not train_mask.any() or not val_mask.any():
            raise ValueError("Empty train/held-out fold")
        y_train, y_val = y[train_mask], y[val_mask]
        if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
            raise ValueError("Both classes needed in every fold")
        pos = int(np.sum(y_train))
        class_weight = float((len(y_train) - pos) / pos)
        for name, (columns, matrix) in variants.items():
            target = work / f"fold{fold_num}_{name}.json"
            if target.is_file():
                previous = json.loads(target.read_text(encoding="utf-8"))
                if previous.get("run_id") == run_id and previous.get("variant") == name:
                    print(f"SKIP fold {fold_num} {name}: checkpoint", flush=True)
                    completed.append(previous)
                    continue
            print(f"TRAIN fold {fold_num}/{args.folds} {name}: "
                  f"{int(train_mask.sum()):,} train / {int(val_mask.sum()):,} val rows",
                  flush=True)
            frame_train = pd.DataFrame(matrix[train_mask], columns=columns, copy=False)
            frame_val = pd.DataFrame(matrix[val_mask], columns=columns, copy=False)
            model = XGBClassifier(**PARAMS, scale_pos_weight=class_weight,
                                  random_state=args.seed)
            model.fit(frame_train, y_train)
            scores = model.predict_proba(frame_val)[:, 1]
            entry = {
                "run_id": run_id, "fold": fold_num, "variant": name,
                "train_patients": len(fold_train_ids), "fold_validation_patients": len(fold_val_ids),
                "train_rows": int(train_mask.sum()), "fold_validation_rows": int(val_mask.sum()),
                "features": len(columns), "scale_pos_weight": class_weight,
                "positive_rate": float(y_val.mean()),
                "AUROC": float(roc_auc_score(y_val, scores)),
                "AP": float(average_precision_score(y_val, scores)),
                "external_validation_patients_evaluated": 0, "test_patients_evaluated": 0,
            }
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(entry, indent=2), encoding="utf-8")
            temp.replace(target)
            completed.append(entry)
            print(f"  AUROC={entry['AUROC']:.4f} AP={entry['AP']:.4f}; checkpoint saved",
                  flush=True)

    table = pd.DataFrame(completed).sort_values(["fold", "variant"])
    report = (table.groupby("variant")[["AUROC", "AP"]]
              .agg(["mean", "std"]).round(6))
    table.to_csv(work / "fold_metrics.csv", index=False)
    report.to_csv(work / "cv_summary.csv")
    summary = {
        "run_id": run_id, "cohort": "locked_train_only",
        "train_patients": len(train_ids), "folds": args.folds,
        "validation_patients_evaluated": 0, "test_patients_evaluated": 0,
        "mean_and_sd": {
            name: {"AUROC_mean": float(part["AUROC"].mean()),
                   "AUROC_std": float(part["AUROC"].std(ddof=1)),
                   "AP_mean": float(part["AP"].mean()),
                   "AP_std": float(part["AP"].std(ddof=1))}
            for name, part in table.groupby("variant")
        },
        "warning": ("Age features were previously selected on fixed Validation. "
                    "This CV is a robustness check, not independent evidence of "
                    "generalization; final Test remains held out."),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report.to_string(), flush=True)
    print(f"Saved: {work / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
