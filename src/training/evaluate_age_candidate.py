"""Evaluate 176-feature XGBoost candidate on locked Validation (no retraining).

Outputs source-wise AUROC/AP, feature gain and ranking, and exploratory alert
burden by thresholds. Scores are NOT calibrated probabilities; thresholds are
diagnostic only and NOT clinical alert policy.

Run: python src/training/evaluate_age_candidate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS

AGE_INDICES = [EXTRA_NAMES.index(f"{v}_age_hours") for v in VARS]
AGE_NAMES = [f"{v}_age_hours" for v in VARS]
MODEL_PATH = Path("models/xgboost_ablation_full_age_only.joblib")
OUT = Path("reports")


def score(y, p):
    if len(y) == 0:
        raise ValueError("Empty scoring group")
    return {
        "rows": int(len(y)), "positive_rate": float(np.mean(y)),
        "AUROC": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "AP": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None,
    }


def main():
    manifest_path = MANIFEST
    if not manifest_path.is_file() or manifest_path.stat().st_size == 0:
        raise FileNotFoundError("Missing locked split manifest")
    cohorts = json.loads(manifest_path.read_text(encoding="utf-8"))["split"]
    if len(set(cohorts["train"] + cohorts["validation"] + cohorts["test"])) != sum(
            len(cohorts[k]) for k in ("train", "validation", "test")):
        raise ValueError("Overlapping patients in manifest")
    index = pd.read_csv(INDEX).set_index("patient_id")
    val_ids = cohorts["validation"]
    if len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicated Validation IDs")
    paths = [Path(index.loc[pid, "path"]) for pid in val_ids]
    base, y, patient_ids = load_or_build("validation", val_ids, paths, CACHE_DIR, False)
    extras = chunks_for("validation", val_ids, paths, CACHE_DIR, False)
    if len(base) != len(extras) or len(y) != len(patient_ids):
        raise ValueError("Validation feature alignment mismatch")
    if not set(patient_ids).issubset(set(val_ids)):
        raise ValueError("Unexpected patient in Validation cache")
    expected_features = FEATURES + AGE_NAMES
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}; run ablate_missingness.py first")
    saved = joblib.load(MODEL_PATH)
    if saved["features"] != expected_features:
        raise ValueError("Model feature schema differs from expected 176 features")
    frame = pd.DataFrame(np.column_stack((base, extras[:, AGE_INDICES])),
                         columns=expected_features)
    model = saved["model"]
    predictions = model.predict_proba(frame)[:, 1]
    summary = {"model": str(MODEL_PATH), "validation_patients": len(val_ids),
               "test_patients_evaluated": 0, "overall": score(y, predictions),
               "by_source": {}, "threshold_metrics": [],
               "warning": ("Thresholds shown for exploration only. Raw score not calibrated; "
                           "row-level FP counts are not counts of distinct clinical alarms. "
                           "Feature gain is non-causal; no Test or utility evaluation.")}
    sources = index["source"]
    for source in sorted(index.loc[val_ids, "source"].unique()):
        source_ids = set(index.index[sources == source]) & set(val_ids)
        keep = np.isin(patient_ids, list(source_ids))
        summary["by_source"][source] = score(y[keep], predictions[keep])

    # Alert-related metrics at descriptive score cutoffs; no decision optimization.
    for cutoff in (0.05, 0.10, 0.20, 0.30, 0.50):
        pred = predictions >= cutoff
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        flagged = set(patient_ids[pred])
        summary["threshold_metrics"].append({
            "score_threshold": cutoff,
            "true_positive_rows": int(tp), "false_positive_rows": int(fp),
            "false_negative_rows": int(fn), "true_negative_rows": int(tn),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "flagged_rows": int(pred.sum()),
            "flagged_patients": len(flagged),
        })

    raw_gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame([
        {"feature": name, "gain": float(raw_gain.get(name, raw_gain.get(f"f{i}", 0.0))),
         "is_age": name in AGE_NAMES,
         "validation_missing_fraction": float(frame[name].isna().mean())}
        for i, name in enumerate(expected_features)
    ]).sort_values("gain", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUT / "xgboost_age176_feature_gain.csv", index=False)
    pd.DataFrame(summary["threshold_metrics"]).to_csv(
        OUT / "xgboost_age176_threshold_diagnostics.csv", index=False)
    location = OUT / "xgboost_age176_validation_analysis.json"
    location.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Overall:", summary["overall"])
    print("By source:", summary["by_source"])
    print("Top 15 features:")
    print(importance.head(15).to_string(index=False))
    print("Threshold diagnostics (not clinical thresholds):")
    print(pd.DataFrame(summary["threshold_metrics"]).to_string(index=False))
    print(f"Saved: {location}")
    print("Test remains held out.")


if __name__ == "__main__":
    main()
