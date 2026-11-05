"""AI-4: ablate measurement missingness features without re-extracting 160 statistics.

Compares previously saved 160/208 baselines against three controlled variants:
  age_only        = 160 + 16 last-observation ages
  count24_only    = 160 + 16 observed counts in 24 rows
  count24_count6  = 160 + 32 observed counts in 24 and 6 rows

Outputs are checkpointed after EACH model. Re-running skips completed, matching
experiments. Uses locked Train/Validation only, never Test.

Usage:
  python src/training/ablate_missingness.py --max-train 300 --max-validation 100
  python src/training/ablate_missingness.py
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
VARIANTS = {
    "age_only": ("age_hours",),
    "count24_only": ("observed_24h",),
    "count24_count6": ("observed_24h", "observed_6h"),
}
PARAMETERS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "n_jobs": -1,
}


def reproducibility_key(train_paths, train_ids, val_paths, val_ids,
                        features, seed):
    """Ensure stale experiment checkpoints cannot silently be reused."""
    h = hashlib.sha256()
    h.update(fingerprint(train_paths, train_ids).encode())
    h.update(fingerprint(val_paths, val_ids).encode())
    h.update(Path(__file__).read_bytes())
    h.update(Path(__file__).with_name("experiment_missingness.py").read_bytes())
    h.update(json.dumps({"features": features, "seed": seed,
                         "parameters": PARAMETERS}, sort_keys=True).encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    train_all, val_all, test_all = (manifest[k] for k in ("train", "validation", "test"))
    if sum(map(len, (train_all, val_all, test_all))) != len(set(train_all + val_all + test_all)):
        raise ValueError("Overlapping patient splits")
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Patient index contains duplicates")
    train_ids = subset(train_all, index, args.max_train, args.seed)
    val_ids = subset(val_all, index, args.max_validation, args.seed + 1)
    lookup = index.set_index("patient_id")["path"]
    train_paths = [Path(lookup.loc[pid]) for pid in train_ids]
    val_paths = [Path(lookup.loc[pid]) for pid in val_ids]
    scope = ("full" if len(train_ids) == len(train_all) and len(val_ids) == len(val_all)
             else f"train{len(train_ids)}_val{len(val_ids)}")
    print(f"Scope={scope}; Test HELD OUT", flush=True)

    xtr, ytr, groups_tr = load_or_build(
        "train", train_ids, train_paths, CACHE_DIR, False)
    xva, yva, groups_va = load_or_build(
        "validation", val_ids, val_paths, CACHE_DIR, False)
    if set(groups_tr) & set(groups_va):
        raise ValueError("Patient leakage")
    if not (np.array_equal(np.unique(ytr), [0, 1]) and
            np.array_equal(np.unique(yva), [0, 1])):
        raise ValueError("Training and validation both require 0/1 labels")
    more_tr = chunks_for("train", train_ids, train_paths, CACHE_DIR, False)
    more_va = chunks_for("validation", val_ids, val_paths, CACHE_DIR, False)
    if len(xtr) != len(more_tr) or len(xva) != len(more_va):
        raise ValueError("Rows differ between base features and missingness features")

    base_tr = pd.DataFrame(xtr, columns=FEATURES, copy=False)
    base_va = pd.DataFrame(xva, columns=FEATURES, copy=False)
    ratio = float((len(ytr) - int(np.sum(ytr))) / int(np.sum(ytr)))
    OUT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    results = []

    # Compare with prior reports but do not imply prior models were retrained.
    original = OUT / f"xgboost_missingness_{scope}.json"
    if original.is_file():
        prior = json.loads(original.read_text(encoding="utf-8"))
        if (prior.get("train_patients") == len(train_ids) and
                prior.get("validation_patients") == len(val_ids) and
                prior.get("train_rows") == len(ytr) and
                prior.get("validation_rows") == len(yva)):
            for name, field in (("baseline_160", "baseline_160"),
                                ("all_208", "with_missingness_208")):
                s = prior[field]
                results.append({"experiment": name, "features": 160 if name == "baseline_160" else 208,
                                "AUROC": s["AUROC"], "AP": s["AP"],
                                "positive_rate": s["positive_rate"],
                                "origin": "prior_report"})
        else:
            print("Prior report cohort mismatch; ignoring earlier scores.", flush=True)

    for label, suffixes in VARIANTS.items():
        chosen_names = [name for name in EXTRA_NAMES
                        if any(name.endswith("_" + suffix) for suffix in suffixes)]
        positions = [EXTRA_NAMES.index(name) for name in chosen_names]
        names = FEATURES + chosen_names
        key = reproducibility_key(train_paths, train_ids, val_paths, val_ids,
                                  names, args.seed)
        record = OUT / f"xgboost_ablation_{scope}_{label}.json"
        model_file = MODELS / f"xgboost_ablation_{scope}_{label}.joblib"
        if record.exists() and model_file.exists():
            old = json.loads(record.read_text(encoding="utf-8"))
            if old.get("fingerprint") == key:
                print(f"SKIP {label}: verified checkpoint", flush=True)
                results.append(old["metrics"])
                continue
        extended_tr = pd.DataFrame(more_tr[:, positions], columns=chosen_names)
        extended_va = pd.DataFrame(more_va[:, positions], columns=chosen_names)
        frame_tr = pd.concat([base_tr, extended_tr], axis=1)
        frame_va = pd.concat([base_va, extended_va], axis=1)
        print(f"TRAIN {label}: {len(names)} features ...", flush=True)
        model = XGBClassifier(**PARAMETERS, scale_pos_weight=ratio,
                              random_state=args.seed)
        model.fit(frame_tr, ytr)
        prob = model.predict_proba(frame_va)[:, 1]
        metrics = {"experiment": label, "features": len(names),
                   "AUROC": float(roc_auc_score(yva, prob)),
                   "AP": float(average_precision_score(yva, prob)),
                   "positive_rate": float(np.mean(yva)),
                   "origin": "new_ablation"}
        # Persist model then checkpoint. A crash before JSON makes the run retrain.
        joblib.dump({"model": model, "features": names, "window_rows": 24,
                     "fingerprint": key, "metrics": metrics}, model_file)
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "scope": scope, "fingerprint": key, "metrics": metrics,
            "train_patients": len(train_ids), "validation_patients": len(val_ids),
            "test_patients_evaluated": 0,
        }, indent=2), encoding="utf-8")
        tmp.replace(record)
        print(f"  AUROC={metrics['AUROC']:.4f} AP={metrics['AP']:.4f}; saved checkpoint",
              flush=True)
        results.append(metrics)

    comparison = pd.DataFrame(results)
    csv = OUT / f"xgboost_ablation_{scope}.csv"
    comparison.to_csv(csv, index=False)
    summary = OUT / f"xgboost_ablation_{scope}.json"
    summary.write_text(json.dumps({
        "scope": scope, "train_patients": len(train_ids),
        "validation_patients": len(val_ids), "test_patients_evaluated": 0,
        "results": results,
        "notes": ("Controlled feature ablations on fixed held-out Validation. "
                  "Prior baseline/208 metrics are read from prior report; "
                  "not a five-fold assessment, not a Test evaluation."),
    }, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)
    print(f"Saved: {csv} and {summary}; Test HELD OUT", flush=True)


if __name__ == "__main__":
    main()
