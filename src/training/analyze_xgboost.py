"""Analyze locked-split XGBoost model without retraining or touching Test.

Run:
  python src/training/analyze_xgboost.py
  python src/training/analyze_xgboost.py --max-train 300 --max-validation 100
Requires cached features and saved compare_xgboost best-AP model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, fingerprint
from train_xgboost import FEATURES, VARS
from train_xgboost_manifest import subset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    index = pd.read_csv(INDEX).set_index("patient_id")
    train_ids = subset(manifest["train"], index.reset_index(), args.max_train, args.seed)
    val_ids = subset(manifest["validation"], index.reset_index(), args.max_validation, args.seed + 1)
    scope = ("full" if len(train_ids) == len(manifest["train"]) and
             len(val_ids) == len(manifest["validation"])
             else f"train{len(train_ids)}_val{len(val_ids)}")
    paths = [Path(index.loc[pid, "path"]) for pid in val_ids]
    key = fingerprint(paths, val_ids)
    cache_path = CACHE_DIR / f"validation_{len(val_ids)}_{key[:16]}.joblib"
    model_path = Path("models") / f"xgboost_compare_{scope}_best_ap.joblib"
    if not cache_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Missing cache or model. Run compare_xgboost.py first.")
    cached = joblib.load(cache_path)
    artifact = joblib.load(model_path)
    if cached["fingerprint"] != key or cached["features"] != FEATURES:
        raise ValueError("Feature cache mismatch; cannot evaluate")
    if artifact["features"] != FEATURES:
        raise ValueError("Model feature schema mismatch")

    x = pd.DataFrame(cached["x"], columns=FEATURES)
    y = np.asarray(cached["y"])
    groups = np.asarray(cached["patient_ids"])
    if not set(groups).issubset(set(val_ids)) or len(groups) != len(y):
        raise ValueError("Unexpected patient IDs or cache length")
    model = artifact["model"]
    predictions = model.predict_proba(x)[:, 1]

    # Gain importance is an internal split contribution, not clinical causation.
    raw_gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame([
        {"feature": name, "gain": float(raw_gain.get(name, raw_gain.get(f"f{i}", 0.0))),
         "missing_fraction": float(x[name].isna().mean())}
        for i, name in enumerate(FEATURES)
    ]).sort_values("gain", ascending=False)
    variable_gain = (importance.assign(variable=importance["feature"].str.rsplit("_", n=1).str[0])
                     .groupby("variable", as_index=False)["gain"].sum()
                     .sort_values("gain", ascending=False))

    def scores(mask: np.ndarray) -> dict:
        yy, ss = y[mask], predictions[mask]
        return {"rows": int(len(yy)), "positive_rate": float(np.mean(yy)),
                "AUROC": float(roc_auc_score(yy, ss)) if len(np.unique(yy)) > 1 else None,
                "AP": float(average_precision_score(yy, ss)) if len(np.unique(yy)) > 1 else None}

    by_source = {}
    for source in sorted(set(index.loc[list(set(groups)), "source"])):
        source_patients = set(index.index[index["source"] == source]) & set(val_ids)
        mask = np.isin(groups, list(source_patients))
        if mask.any():
            by_source[source] = scores(mask)

    # Precision/recall points for later alert threshold analysis; illustrative only.
    p, r, thresholds = precision_recall_curve(y, predictions)
    curve = pd.DataFrame({"threshold": thresholds, "precision": p[:-1], "recall": r[:-1]})
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(report_dir / f"xgboost_importance_{scope}.csv", index=False)
    variable_gain.to_csv(report_dir / f"xgboost_variable_gain_{scope}.csv", index=False)
    curve.to_csv(report_dir / f"xgboost_pr_curve_{scope}.csv", index=False)
    summary = {"scope": scope, "model": str(model_path), "patients": len(val_ids),
               "overall": scores(np.ones(len(y), dtype=bool)),
               "by_source": by_source, "top_features": importance.head(20).to_dict("records"),
               "caution": ("Gain is not causal. Validation only; score not calibrated. "
                           "No Test predictions, Group 5-Fold, SHAP, or clinical utility.")}
    path = report_dir / f"xgboost_analysis_{scope}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Overall:", summary["overall"])
    print("By source:", by_source)
    print("Top 10 features:")
    print(importance.head(10).to_string(index=False))
    print(f"Saved: {path}")
    print("Test remains held out.")


if __name__ == "__main__":
    main()
