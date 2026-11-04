"""AI-4: locked-split XGBoost ablations with reusable, validated feature caches.

Quick check:
  python src/training/compare_xgboost.py --max-train 300 --max-validation 100
Full-data cache (first run extracts features; later runs reuse them):
  python src/training/compare_xgboost.py --cache-only
Full-data controlled comparison:
  python src/training/compare_xgboost.py

Test patients are never loaded, fitted, or scored.
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

from train_xgboost import FEATURES, VARS, patient_rows
from train_xgboost_manifest import subset

MANIFEST = Path("data/splits/patient_splits.json")
INDEX = Path("data/splits/patient_index.csv")
CACHE_DIR = Path("data/processed/xgboost_cache")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
FEATURE_SOURCE = Path(__file__).with_name("train_xgboost.py")
CACHE_FORMAT = 1


def fingerprint(paths: list[Path], ids: list[str]) -> str:
    """Invalidate on changed patient membership, raw files, or feature code."""
    h = hashlib.sha256()
    h.update(str(CACHE_FORMAT).encode())
    h.update(FEATURE_SOURCE.read_bytes())
    h.update(json.dumps(FEATURES).encode())
    for patient_id, path in zip(ids, paths):
        meta = path.stat()
        h.update(f"{patient_id}|{path.as_posix()}|{meta.st_size}|{meta.st_mtime_ns}\n".encode())
    return h.hexdigest()


def load_or_build(split: str, ids: list[str], paths: list[Path], root: Path,
                  rebuild: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Save only existing raw-observation 160 features; no imputation or test data."""
    if len(ids) != len(paths) or not ids:
        raise ValueError("Patient IDs and paths must agree and be nonempty")
    key = fingerprint(paths, ids)
    target = root / f"{split}_{len(ids)}_{key[:16]}.joblib"
    if target.is_file() and not rebuild:
        print(f"Cache HIT: {target}", flush=True)
        data = joblib.load(target)
        if data.get("fingerprint") != key or data.get("features") != FEATURES:
            raise ValueError(f"Cache metadata mismatch: {target}")
        return data["x"], data["y"], data["patient_ids"]

    print(f"Cache MISS: extracting {split} ({len(ids):,} patients)", flush=True)
    xs: list[list[float]] = []
    ys: list[int] = []
    groups: list[str] = []
    for i, (patient_id, path) in enumerate(zip(ids, paths), 1):
        vectors, labels = patient_rows(path)
        xs.extend(vectors)
        ys.extend(labels)
        groups.extend([patient_id] * len(labels))
        if i % 1000 == 0:
            print(f"  {split} {i:,}/{len(ids):,}", flush=True)
    # float64 retains the numerical precision of the existing baseline DataFrame.
    x = np.asarray(xs, dtype=np.float64).reshape(-1, len(FEATURES))
    y = np.asarray(ys, dtype=np.int8)
    group_ids = np.asarray(groups, dtype=str)
    if x.shape[0] != y.shape[0] or y.shape[0] != group_ids.shape[0]:
        raise AssertionError("Cache rows/labels/patient IDs differ")
    if not np.isin(y, [0, 1]).all() or np.isinf(x).any():
        raise ValueError("Invalid cached labels or infinite features")
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump({"fingerprint": key, "features": FEATURES, "x": x,
                 "y": y, "patient_ids": group_ids}, target, compress=3)
    print(f"Cache saved: {target} ({len(y):,} samples)", flush=True)
    return x, y, group_ids


def evaluate(y: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(y, scores)),
        "AP": float(average_precision_score(y, scores)),
        "positive_rate": float(np.mean(y)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=MANIFEST)
    parser.add_argument("--patient-index", type=Path, default=INDEX)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split = manifest["split"]
    train_all, val_all, test_all = (split[k] for k in ("train", "validation", "test"))
    if len(set(train_all + val_all + test_all)) != sum(map(len, (train_all, val_all, test_all))):
        raise ValueError("Overlapping patient IDs in locked manifest")
    index = pd.read_csv(args.patient_index)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs in index")

    train_ids = subset(train_all, index, args.max_train, args.seed)
    val_ids = subset(val_all, index, args.max_validation, args.seed + 1)
    path_by_id = index.set_index("patient_id")["path"]
    train_paths = [Path(path_by_id.loc[pid]) for pid in train_ids]
    val_paths = [Path(path_by_id.loc[pid]) for pid in val_ids]
    print(f"Locked split: train {len(train_ids):,}/{len(train_all):,}, "
          f"validation {len(val_ids):,}/{len(val_all):,}, Test HELD OUT", flush=True)

    x_train, y_train, train_groups = load_or_build(
        "train", train_ids, train_paths, args.cache_dir, args.rebuild_cache)
    x_val, y_val, val_groups = load_or_build(
        "validation", val_ids, val_paths, args.cache_dir, args.rebuild_cache)
    if set(train_groups) & set(val_groups):
        raise AssertionError("Patient leakage between training and validation")
    if args.cache_only:
        print("Caches ready. No model fitting or Test scoring.")
        return
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both training and validation must have positive/negative labels")

    ratio = float((len(y_train) - int(y_train.sum())) / int(y_train.sum()))
    # One factor changes at a time: baseline vs depth vs class weighting.
    configurations = [
        ("baseline", {"max_depth": 4, "scale_pos_weight": ratio}),
        ("depth_3", {"max_depth": 3, "scale_pos_weight": ratio}),
        ("depth_6", {"max_depth": 6, "scale_pos_weight": ratio}),
        ("weight_1", {"max_depth": 4, "scale_pos_weight": 1.0}),
    ]
    scope = ("full" if len(train_ids) == len(train_all) and len(val_ids) == len(val_all)
             else f"train{len(train_ids)}_val{len(val_ids)}")
    results = []
    frame_train = pd.DataFrame(x_train, columns=FEATURES, copy=False)
    frame_val = pd.DataFrame(x_val, columns=FEATURES, copy=False)
    best_ap = -1.0
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for name, overrides in configurations:
        model = XGBClassifier(
            n_estimators=200, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", eval_metric="logloss",
            n_jobs=-1, random_state=args.seed, **overrides,
        )
        print(f"Training {name} ...", flush=True)
        model.fit(frame_train, y_train)
        predictions = model.predict_proba(frame_val)[:, 1]
        scores = evaluate(y_val, predictions)
        entry = {"experiment": name, "train_patients": len(train_ids),
                 "validation_patients": len(val_ids), "train_samples": len(y_train),
                 "validation_samples": len(y_val), "test_patients_evaluated": 0,
                 "max_depth": overrides["max_depth"],
                 "scale_pos_weight": overrides["scale_pos_weight"], **scores}
        results.append(entry)
        print(f"  AUROC={scores['AUROC']:.4f} AP={scores['AP']:.4f} "
              f"positive_rate={scores['positive_rate']:.4f}", flush=True)
        if scores["AP"] > best_ap:
            best_ap = scores["AP"]
            output = MODEL_DIR / f"xgboost_compare_{scope}_best_ap.joblib"
            joblib.dump({"model": model, "features": FEATURES,
                         "selected_columns": VARS, "window_rows": 24,
                         "experiment": entry}, output)
    table = pd.DataFrame(results)
    csv_path = REPORT_DIR / f"xgboost_compare_{scope}.csv"
    json_path = REPORT_DIR / f"xgboost_compare_{scope}.json"
    table.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps({
        "description": "Single Validation split ablations; NOT 5-fold or final Test evaluation",
        "scope": scope, "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved comparison: {csv_path} and {json_path}")
    print("Best-AP candidate saved separately. Selection is provisional until Group 5-Fold.")
    print("Test data not loaded or evaluated.")


if __name__ == "__main__":
    main()
