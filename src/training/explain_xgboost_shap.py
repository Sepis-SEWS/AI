"""AI-6: explain fixed XGBoost-176 candidate using native TreeSHAP.

Native XGBoost pred_contribs returns TreeSHAP-like additive attributions in
RAW MARGIN (log-odds), including bias. These are NOT clinical causal effects.
Uses the locked Validation cache only (no retraining and NO Test access).

Usage (from project root):
  python src/training/explain_xgboost_shap.py --max-rows 1000
  python src/training/explain_xgboost_shap.py --max-rows 3000 --seed 42

Output is an aggregate per-feature explanation and per-Set A/B comparison.
No individual patient-level attributions are exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS

MODEL = Path("models/xgboost_ablation_full_age_only.joblib")
OUTPUT = Path("reports")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_rows < 100:
        parser.error("--max-rows must be at least 100")
    cohorts = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if len(set(cohorts["train"] + cohorts["validation"] + cohorts["test"])) != sum(
            len(cohorts[k]) for k in ("train", "validation", "test")):
        raise ValueError("Patient manifest has overlapping splits")
    ids = cohorts["validation"]
    index = pd.read_csv(INDEX).set_index("patient_id")
    paths = [Path(index.loc[pid, "path"]) for pid in ids]
    base, labels, patient_ids = load_or_build("validation", ids, paths, CACHE_DIR, False)
    extra = chunks_for("validation", ids, paths, CACHE_DIR, False)
    if not (len(base) == len(extra) == len(labels) == len(patient_ids)):
        raise ValueError("Misaligned feature arrays")
    if not set(patient_ids).issubset(set(ids)):
        raise ValueError("Unknown patient in Validation cache")
    age_names = [f"{var}_age_hours" for var in VARS]
    names = FEATURES + age_names
    if len(names) != 176:
        raise ValueError("Expected 176 features")
    saved = joblib.load(MODEL)
    if saved.get("features") != names:
        raise ValueError("XGBoost model feature schema mismatch")
    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(len(labels), size=min(args.max_rows, len(labels)),
                                  replace=False))
    all_age_cols = [EXTRA_NAMES.index(name) for name in age_names]
    frames = pd.DataFrame(
        np.column_stack((base[selected], extra[selected][:, all_age_cols])),
        columns=names)
    booster = saved["model"].get_booster()
    contributions = booster.predict(
        xgb.DMatrix(frames, feature_names=names), pred_contribs=True)
    if contributions.shape != (len(selected), len(names) + 1):
        raise ValueError("Unexpected SHAP contribution dimensions")
    raw_margin = booster.predict(
        xgb.DMatrix(frames, feature_names=names), output_margin=True)
    if not np.allclose(contributions.sum(axis=1), raw_margin, rtol=1e-4, atol=1e-4):
        raise AssertionError("SHAP additive attributions do not reproduce model margin")

    abs_values = np.abs(contributions[:, :-1])
    source_by_patient = index["source"]
    source = np.array([source_by_patient.loc[pid] for pid in patient_ids[selected]])
    rows = []
    for column, name in enumerate(names):
        entry = {
            "feature": name,
            "is_age_feature": name in age_names,
            "n_sampled_rows": len(selected),
            "mean_abs_shap_raw_logodds": float(abs_values[:, column].mean()),
            "mean_shap_raw_logodds": float(contributions[:, column].mean()),
        }
        for group in ("training_setA", "training_setB"):
            mask = source == group
            entry[f"{group}_mean_abs_shap"] = (
                float(abs_values[mask, column].mean()) if mask.any() else None)
        rows.append(entry)
    table = pd.DataFrame(rows).sort_values(
        "mean_abs_shap_raw_logodds", ascending=False)
    OUTPUT.mkdir(exist_ok=True)
    destination = OUTPUT / "xgboost_age176_shap_importance.csv"
    table.to_csv(destination, index=False)
    report = {
        "scope": "locked_validation_sample_only",
        "rows_sampled": len(selected),
        "validation_rows_total": len(labels),
        "test_patients_evaluated": 0,
        "seed": args.seed,
        "unit": "raw XGBoost margin (log odds), NOT probability percentage",
        "additivity_verified": True,
        "total_absolute_shap_by_age_features":
            float(abs_values[:, [names.index(n) for n in age_names]].sum() /
                  abs_values.sum()) if abs_values.sum() else None,
        "top_20": table.head(20).to_dict(orient="records"),
        "warning": (
            "TreeSHAP attributions describe the fitted model, not causation. "
            "FiO2 measurement timing may proxy source-specific clinical "
            "measurement practices. Validation sampling is descriptive and "
            "not final independent evaluation."
        ),
    }
    summary_path = OUTPUT / "xgboost_age176_shap_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(table.head(20).to_string(index=False), flush=True)
    print(f"Age-feature fraction of absolute SHAP: "
          f"{report['total_absolute_shap_by_age_features']:.4f}", flush=True)
    print(f"Saved: {destination}, {summary_path}; Test HELD OUT", flush=True)


if __name__ == "__main__":
    main()
