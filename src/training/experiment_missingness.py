"""AI-4 causal missingness-feature ablation against the unchanged 160-feature XGBoost.

Usage:
  python src/training/experiment_missingness.py --max-train 300 --max-validation 100
  python src/training/experiment_missingness.py

Adds 3 features for each of the 16 existing variables:
  *_observed_24h: count of finite observations in the current/previous 23 rows
  *_observed_6h: count in the most recent six rows
  *_age_hours: elapsed ICU hours since last finite observation, capped at 48,
                with 48 indicating never seen within the window
No future records, interpolation, Test data, or target-derived features.
Extracts missingness metadata separately and checkpoints every 250 patients.
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

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, load_or_build
from train_xgboost import FEATURES, VARS
from train_xgboost_manifest import subset

EXTRA_NAMES = [f"{v}_{suffix}" for v in VARS
               for suffix in ("observed_24h", "observed_6h", "age_hours")]
CHUNK_SIZE = 250


def patient_missingness(path: Path) -> np.ndarray:
    """Return one (N-23, 48) causal matrix in the same row order as patient_rows."""
    df = pd.read_csv(path, sep="|")
    df = df.sort_values("ICULOS", kind="stable").reset_index(drop=True)
    if any(v not in df.columns for v in VARS) or "ICULOS" not in df:
        raise ValueError(f"{path}: required columns missing")
    n = len(df)
    if n < 24:
        return np.empty((0, len(EXTRA_NAMES)), dtype=np.float32)
    times = pd.to_numeric(df["ICULOS"], errors="coerce").to_numpy(dtype=np.float64)
    values = df[VARS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(times)[:, None]
    # One causal output for every 24-row history ending at index >=23.
    windows = np.lib.stride_tricks.sliding_window_view(valid, 24, axis=0)
    # shape from sliding_window_view is (N-23, 16, 24)
    observed24 = windows.sum(axis=2).astype(np.float32)
    observed6 = windows[:, :, -6:].sum(axis=2).astype(np.float32)
    rev = windows[:, :, ::-1]
    seen = observed24 > 0
    back = np.argmax(rev, axis=2)
    last_idx = np.arange(23, n)[:, None] - back
    last_times = times[last_idx]
    age = np.broadcast_to(times[23:, None], last_idx.shape) - last_times
    age = np.where(seen, np.clip(age, 0, 48), 48).astype(np.float32)
    stacked = np.stack((observed24, observed6, age), axis=2)
    return stacked.reshape(n - 23, len(EXTRA_NAMES))


def chunks_for(split: str, ids: list[str], paths: list[Path], root: Path,
               rebuild: bool) -> np.ndarray:
    """Checkpoint missingness extraction per chunk; reuse after interruption."""
    h = hashlib.sha256()
    h.update(Path(__file__).read_bytes())
    for pid, path in zip(ids, paths):
        st = path.stat()
        h.update(f"{pid}|{path.as_posix()}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    key = h.hexdigest()[:16]
    folder = root / f"missingness_{split}_{len(ids)}_{key}"
    folder.mkdir(parents=True, exist_ok=True)
    arrays = []
    for start in range(0, len(ids), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(ids))
        target = folder / f"{start:06d}_{stop:06d}.joblib"
        if target.exists() and not rebuild:
            chunk = joblib.load(target)
        else:
            pieces = [patient_missingness(path) for path in paths[start:stop]]
            chunk = np.concatenate(pieces, axis=0)
            temp = target.with_suffix(".tmp")
            joblib.dump(chunk, temp, compress=3)
            temp.replace(target)
            print(f"Saved {split} missingness chunk {stop:,}/{len(ids):,}", flush=True)
        if chunk.ndim != 2 or chunk.shape[1] != len(EXTRA_NAMES):
            raise ValueError(f"Invalid chunk: {target}")
        arrays.append(chunk)
    return np.concatenate(arrays, axis=0)


def run_model(xtrain: pd.DataFrame, ytrain: np.ndarray,
              xval: pd.DataFrame, yval: np.ndarray, seed: int):
    pos = int(ytrain.sum())
    weight = (len(ytrain) - pos) / pos
    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        scale_pos_weight=weight, eval_metric="logloss", n_jobs=-1,
        random_state=seed,
    )
    model.fit(xtrain, ytrain)
    pred = model.predict_proba(xval)[:, 1]
    return model, {"AUROC": float(roc_auc_score(yval, pred)),
                   "AP": float(average_precision_score(yval, pred)),
                   "positive_rate": float(yval.mean())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-missingness", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    index = pd.read_csv(INDEX)
    train_ids = subset(manifest["train"], index, args.max_train, args.seed)
    val_ids = subset(manifest["validation"], index, args.max_validation, args.seed + 1)
    if set(train_ids) & set(val_ids):
        raise ValueError("Patient leakage")
    paths = index.set_index("patient_id")["path"]
    train_paths = [Path(paths.loc[p]) for p in train_ids]
    val_paths = [Path(paths.loc[p]) for p in val_ids]
    scope = "full" if (len(train_ids) == len(manifest["train"]) and
                       len(val_ids) == len(manifest["validation"])) else (
                           f"train{len(train_ids)}_val{len(val_ids)}")
    print(f"Scope={scope}; Test HELD OUT", flush=True)
    xtr, ytr, grp_tr = load_or_build("train", train_ids, train_paths, CACHE_DIR, False)
    xva, yva, grp_va = load_or_build("validation", val_ids, val_paths, CACHE_DIR, False)
    assert not (set(grp_tr) & set(grp_va))
    extra_train = chunks_for("train", train_ids, train_paths, CACHE_DIR,
                             args.rebuild_missingness)
    extra_val = chunks_for("validation", val_ids, val_paths, CACHE_DIR,
                           args.rebuild_missingness)
    if len(xtr) != len(extra_train) or len(xva) != len(extra_val):
        raise ValueError("Missingness and original 160-feature row counts differ")
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        raise ValueError("Both classes are needed")

    base_train = pd.DataFrame(xtr, columns=FEATURES)
    base_val = pd.DataFrame(xva, columns=FEATURES)
    extended_train = pd.concat(
        [base_train, pd.DataFrame(extra_train, columns=EXTRA_NAMES)], axis=1)
    extended_val = pd.concat(
        [base_val, pd.DataFrame(extra_val, columns=EXTRA_NAMES)], axis=1)
    del extra_train, extra_val
    print("Fitting original 160 features ...", flush=True)
    baseline, base_metrics = run_model(base_train, ytr, base_val, yva, args.seed)
    print("Fitting 160 + 48 missingness features ...", flush=True)
    improved, extra_metrics = run_model(
        extended_train, ytr, extended_val, yva, args.seed)
    print("160:", base_metrics, flush=True)
    print("208:", extra_metrics, flush=True)
    report = {
        "scope": scope, "train_patients": len(train_ids),
        "validation_patients": len(val_ids), "test_patients_evaluated": 0,
        "train_rows": len(ytr), "validation_rows": len(yva),
        "baseline_160": base_metrics,
        "with_missingness_208": extra_metrics,
        "delta_auroc": extra_metrics["AUROC"] - base_metrics["AUROC"],
        "delta_ap": extra_metrics["AP"] - base_metrics["AP"],
        "note": ("Same splits/hyperparameters; only 48 causal measurement features change. "
                 "Single Validation split, not Group 5-Fold; not calibrated.")}
    out = Path("reports")
    out.mkdir(exist_ok=True)
    path = out / f"xgboost_missingness_{scope}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    gain = improved.get_booster().get_score(importance_type="gain")
    ranking = pd.DataFrame([
        {"feature": name, "gain": float(gain.get(name, gain.get(f"f{i}", 0.0))),
         "is_missingness": name in EXTRA_NAMES}
        for i, name in enumerate(FEATURES + EXTRA_NAMES)
    ]).sort_values("gain", ascending=False)
    ranking.to_csv(out / f"xgboost_missingness_importance_{scope}.csv", index=False)
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"model": improved, "features": FEATURES + EXTRA_NAMES,
                 "window_rows": 24, "experiment": report},
                model_dir / f"xgboost_missingness_{scope}.joblib")
    print(f"Saved: {path}; Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
