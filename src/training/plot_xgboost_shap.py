"""Create descriptive TreeSHAP figures from LOCKED Validation only.

Run: python src/training/plot_xgboost_shap.py --max-rows 1000
Uses XGBoost native pred_contribs (RAW margin/log-odds). No Test or training.
Plots: global mean absolute SHAP and a representative sampled-window waterfall.
The waterfall is a MODEL explanation, not evidence of causation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS

MODEL = Path("models/xgboost_ablation_full_age_only.joblib")
OUT = Path("reports")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_rows < 100:
        parser.error("--max-rows must be >=100")
    split = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    ids = split["validation"]
    index = pd.read_csv(INDEX).set_index("patient_id")
    paths = [Path(index.loc[p, "path"]) for p in ids]
    base, labels, patients = load_or_build("validation", ids, paths, CACHE_DIR, False)
    extra = chunks_for("validation", ids, paths, CACHE_DIR, False)
    if len(base) != len(extra) or len(labels) != len(patients) or len(base) != len(labels):
        raise ValueError("Misaligned validation cache")
    if not set(patients).issubset(set(ids)):
        raise ValueError("Non-validation patient in cache")
    age_names = [f"{var}_age_hours" for var in VARS]
    names = FEATURES + age_names
    saved = joblib.load(MODEL)
    if saved.get("features") != names:
        raise ValueError("XGBoost feature order changed")
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(len(labels), min(args.max_rows, len(labels)), replace=False))
    age_cols = [EXTRA_NAMES.index(name) for name in age_names]
    frame = pd.DataFrame(
        np.column_stack([base[chosen], extra[chosen][:, age_cols]]), columns=names)
    booster = saved["model"].get_booster()
    dmat = xgb.DMatrix(frame, feature_names=names)
    contributions = booster.predict(dmat, pred_contribs=True)
    margin = booster.predict(dmat, output_margin=True)
    if not np.allclose(contributions.sum(axis=1), margin, atol=1e-4, rtol=1e-4):
        raise ValueError("SHAP additivity failed")
    mean_abs = np.abs(contributions[:, :-1]).mean(axis=0)
    top = np.argsort(mean_abs)[-20:][::-1]
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(np.array(names)[top][::-1], mean_abs[top][::-1])
    ax.set_xlabel("Mean absolute TreeSHAP (raw log-odds)")
    ax.set_title("XGBoost-176: global feature contributions (Validation sample)")
    fig.tight_layout()
    fig.savefig(OUT / "xgboost_age176_shap_summary.png", dpi=170)
    plt.close(fig)

    # Select a reproducible sampled window for a purely illustrative explanation.
    idx = 0
    one = contributions[idx, :-1]
    order = np.argsort(np.abs(one))[-15:][::-1]
    other = float(one.sum() - one[order].sum())
    values = list(one[order]) + [other]
    labels_plot = list(np.asarray(names)[order]) + ["Other 161 features"]
    fig, ax = plt.subplots(figsize=(11, 8))
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos, labels_plot)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("Signed TreeSHAP contribution (raw log-odds)")
    ax.set_title("Illustrative one-window attribution (not patient diagnosis)")
    fig.tight_layout()
    fig.savefig(OUT / "xgboost_age176_shap_local_contributions.png", dpi=170)
    plt.close(fig)
    summary = {
        "scope": "locked_validation_sample_only",
        "sampled_rows": len(chosen),
        "test_evaluated": False,
        "units": "XGBoost raw margin (log-odds)",
        "representative_window_prediction_margin": float(margin[idx]),
        "representative_window_bias": float(contributions[idx, -1]),
        "additivity_verified": True,
        "notes": "Plots are descriptive, non-causal, not probability contributions. Local plot shows a reproducibly selected window without identifying information.",
        "files": [
            "reports/xgboost_age176_shap_summary.png",
            "reports/xgboost_age176_shap_local_contributions.png",
        ],
    }
    (OUT / "xgboost_age176_shap_plots.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved SHAP summary/local contribution figures (Validation only). Test NOT accessed.")


if __name__ == "__main__":
    main()
