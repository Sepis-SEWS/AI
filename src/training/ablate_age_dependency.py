"""AI-4 diagnostic ablation: does 176-feature XGBoost overuse FiO2 age?

Uses the exact locked train/validation cohorts and existing 160-feature cache /
48-feature missingness checkpoints. Train only the *new* candidate variants.
Never reads Test. Each run persists a checkpoint after each candidate.

Run:
  python src/training/ablate_age_dependency.py --max-train 300 --max-validation 100
  python src/training/ablate_age_dependency.py
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
from xgboost import XGBClassifier

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, fingerprint, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS
from train_xgboost_manifest import subset

OUT = Path("reports")
MODELS = Path("models")
AGE_NAMES = [f"{v}_age_hours" for v in VARS]
AGE_COLUMNS = [EXTRA_NAMES.index(n) for n in AGE_NAMES]
PARAMS = {
    "n_estimators": 200, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist",
    "eval_metric": "logloss", "n_jobs": -1,
}
VARIANTS = {
    # Keep all 160 vitals/lab statistics; only test dependence on their age features.
    "age_without_fio2_age": [n for n in AGE_NAMES if n != "FiO2_age_hours"],
    "age_without_lab_ages": [f"{v}_age_hours" for v in
                              ("HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat")],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not MANIFEST.is_file() or not MANIFEST.stat().st_size:
        raise FileNotFoundError("Locked manifest missing. Do not overwrite it.")
    cohorts = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    groups = [cohorts[k] for k in ("train", "validation", "test")]
    if len(set(sum(groups, []))) != sum(map(len, groups)):
        raise ValueError("Overlapping patient cohorts")
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs")
    train_ids = subset(cohorts["train"], index, args.max_train, args.seed)
    val_ids = subset(cohorts["validation"], index, args.max_validation, args.seed + 1)
    paths = index.set_index("patient_id")["path"]
    train_paths = [Path(paths.loc[pid]) for pid in train_ids]
    val_paths = [Path(paths.loc[pid]) for pid in val_ids]
    scope = ("full" if len(train_ids) == len(cohorts["train"]) and
             len(val_ids) == len(cohorts["validation"])
             else f"train{len(train_ids)}_val{len(val_ids)}")
    print(f"Scope={scope}; Test HELD OUT", flush=True)

    xtr, ytr, ptr = load_or_build("train", train_ids, train_paths, CACHE_DIR, False)
    xva, yva, pva = load_or_build("validation", val_ids, val_paths, CACHE_DIR, False)
    if set(ptr) & set(pva):
        raise ValueError("Patient leakage across train and validation")
    extra_tr = chunks_for("train", train_ids, train_paths, CACHE_DIR, False)
    extra_va = chunks_for("validation", val_ids, val_paths, CACHE_DIR, False)
    if not (len(xtr) == len(ytr) == len(extra_tr) == len(ptr)
            and len(xva) == len(yva) == len(extra_va) == len(pva)):
        raise ValueError("Feature rows, labels and patients misaligned")
    if len(np.unique(ytr)) != 2 or len(np.unique(yva)) != 2:
        raise ValueError("Both classes required")
    ratio = (len(ytr) - int(ytr.sum())) / int(ytr.sum())
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for name, ages in VARIANTS.items():
        selected = [EXTRA_NAMES.index(col) for col in ages]
        columns = FEATURES + ages
        h = hashlib.sha256()
        h.update(Path(__file__).read_bytes())
        h.update(Path(__file__).with_name("experiment_missingness.py").read_bytes())
        h.update(fingerprint(train_paths, train_ids).encode())
        h.update(fingerprint(val_paths, val_ids).encode())
        h.update(json.dumps({"name": name, "columns": columns,
                             "params": PARAMS, "seed": args.seed}, sort_keys=True).encode())
        key = h.hexdigest()
        checkpoint = OUT / f"xgboost_age_dependency_{scope}_{name}.json"
        model_file = MODELS / f"xgboost_age_dependency_{scope}_{name}.joblib"
        if checkpoint.is_file() and model_file.is_file():
            old = json.loads(checkpoint.read_text(encoding="utf-8"))
            if old.get("fingerprint") == key:
                print(f"SKIP {name}: verified checkpoint", flush=True)
                results.append(old["metrics"])
                continue

        frame_tr = pd.DataFrame(
            np.column_stack((xtr, extra_tr[:, selected])), columns=columns)
        frame_va = pd.DataFrame(
            np.column_stack((xva, extra_va[:, selected])), columns=columns)
        print(f"TRAIN {name} ({len(columns)} features) ...", flush=True)
        model = XGBClassifier(**PARAMS, scale_pos_weight=float(ratio),
                              random_state=args.seed)
        model.fit(frame_tr, ytr)
        pred = model.predict_proba(frame_va)[:, 1]
        metrics = {
            "experiment": name, "features": len(columns),
            "AUROC": float(roc_auc_score(yva, pred)),
            "AP": float(average_precision_score(yva, pred)),
            "positive_rate": float(np.mean(yva)),
            "test_patients_evaluated": 0,
        }
        joblib.dump({"model": model, "features": columns,
                     "fingerprint": key, "experiment": metrics}, model_file)
        tmp = checkpoint.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": key, "metrics": metrics,
                                    "scope": scope, "train_patients": len(train_ids),
                                    "validation_patients": len(val_ids)}, indent=2),
                       encoding="utf-8")
        tmp.replace(checkpoint)
        results.append(metrics)
        print(f"  AUROC={metrics['AUROC']:.4f} AP={metrics['AP']:.4f}; checkpoint saved",
              flush=True)

    # Report reference numbers only for identical full fixed validation cohort.
    if scope == "full":
        results.insert(0, {"experiment": "age_176_reference",
                           "features": 176, "AUROC": 0.7900623071762783,
                           "AP": 0.09487981162898537,
                           "positive_rate": float(np.mean(yva)),
                           "test_patients_evaluated": 0,
                           "origin": "prior_validation_result"})
    df = pd.DataFrame(results)
    report = OUT / f"xgboost_age_dependency_{scope}.csv"
    df.to_csv(report, index=False)
    print(df.to_string(index=False), flush=True)
    print(f"Saved: {report}; Test HELD OUT", flush=True)
    print("Importance is diagnostic only: high gain is not proof of spurious leakage.")


if __name__ == "__main__":
    main()
