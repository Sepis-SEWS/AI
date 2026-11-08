"""Produce remaining descriptive SEWS Validation diagnostics without Test access.

Usage: python src/training/report_validation_diagnostics.py

Reads already-aligned Validation scores and produces PR curves, reliability
diagrams, sensitivity at >=95% specificity, and classification reports at the
FROZEN 0.2 threshold. This does not change the locked Test policy, fit
anything, or re-evaluate Test. Curves are on 24-row-scoreable Validation hours.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, precision_recall_curve, roc_curve)

ARCHIVE = Path("reports/ensemble_validation_scores.npz")
OUT = Path("reports")
FROZEN_THRESHOLD = 0.2


def main() -> None:
    policy_path = OUT / "final_evaluation_policy_locked.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if (policy.get("status") != "LOCKED_BEFORE_TEST" or
            policy.get("model") != "lstm_full_pw10" or
            float(policy.get("score_threshold")) != FROZEN_THRESHOLD):
        raise ValueError("Locked policy not found or changed")
    with np.load(ARCHIVE, allow_pickle=False) as f:
        y = f["labels"].astype(int)
        pid = f["patient_ids"]
        xgb = f["xgboost_scores"].astype(float)
        lstm = f["lstm_scores"].astype(float)
    if not (len(y) == len(pid) == len(xgb) == len(lstm)) or not len(y):
        raise ValueError("Validation score archive is not aligned")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid labels")

    models = {
        "XGBoost 176": xgb,
        "LSTM pw10": lstm,
        "Ensemble 25:75": 0.25 * xgb + 0.75 * lstm,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    fig_cal, ax_cal = plt.subplots(figsize=(8, 6))
    diagnostics = []
    for name, p in models.items():
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError(f"Invalid scores for {name}")
        precision, recall, _ = precision_recall_curve(y, p)
        ax_pr.plot(recall, precision, label=f"{name} (AP={average_precision_score(y, p):.3f})")
        fpr, tpr, thresholds = roc_curve(y, p)
        eligible = np.flatnonzero(fpr <= 0.05 + 1e-12)
        pick = eligible[np.argmax(tpr[eligible])]
        selected = p >= FROZEN_THRESHOLD
        matrix = confusion_matrix(y, selected, labels=[0, 1])
        report = classification_report(y, selected, labels=[0, 1],
                                       output_dict=True, zero_division=0)
        n_bins = 10
        boundaries = np.linspace(0, 1, n_bins + 1)
        group = np.searchsorted(boundaries[1:-1], p, side="right")
        observed, predicted, counts = [], [], []
        for k in range(n_bins):
            mask = group == k
            if mask.any():
                observed.append(float(y[mask].mean()))
                predicted.append(float(p[mask].mean()))
                counts.append(int(mask.sum()))
        ax_cal.plot(predicted, observed, "o-", label=name)
        diagnostics.append({
            "model": name,
            "validation_scoreable_rows": int(len(y)),
            "validation_scored_patients": int(len(set(pid))),
            "sensitivity_at_specificity_at_least_95pct": float(tpr[pick]),
            "specificity_at_selected_roc_point": float(1 - fpr[pick]),
            "diagnostic_roc_score_threshold": (float(thresholds[pick])
                                                if np.isfinite(thresholds[pick]) else None),
            "frozen_threshold_for_classification_report": FROZEN_THRESHOLD,
            "confusion_matrix_0_1": matrix.tolist(),
            "classification_report": report,
            "reliability_bin_mean_prediction": predicted,
            "reliability_bin_observed_fraction": observed,
            "reliability_bin_counts": counts,
        })
    ax_pr.set(xlabel="Recall", ylabel="Precision",
              title="Locked Validation: Precision-Recall (scoreable hours)")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1)
    ax_pr.legend()
    fig_pr.tight_layout()
    fig_pr.savefig(OUT / "validation_precision_recall_curve.png", dpi=160)
    plt.close(fig_pr)
    ax_cal.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax_cal.set(xlabel="Mean raw model score", ylabel="Observed positive frequency",
               title="Locked Validation: reliability (10 equal-width bins)")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.legend()
    fig_cal.tight_layout()
    fig_cal.savefig(OUT / "validation_reliability_diagram.png", dpi=160)
    plt.close(fig_cal)
    target = OUT / "validation_required_metrics.json"
    target.write_text(json.dumps({
        "scope": "Validation 24-row-scoreable timestamps only",
        "frozen_test_policy_unchanged": True,
        "test_evaluated_in_this_run": False,
        "warnings": [
            "Sensitivity at 95% specificity is descriptive, not a new Test cutoff.",
            "Uncalibrated weighted model scores are not probabilities.",
            "The final Test is NEVER reread; all diagrams come from Validation archive.",
        ], "models": diagnostics,
    }, indent=2), encoding="utf-8")
    for entry in diagnostics:
        print(entry["model"],
              f"Sensitivity@>=95%Specificity={entry['sensitivity_at_specificity_at_least_95pct']:.4f}",
              f"Frozen threshold={FROZEN_THRESHOLD}",
              "ConfusionMatrix=", entry["confusion_matrix_0_1"], flush=True)
    print(f"Saved: {target} and two PNG charts. Test NOT accessed.", flush=True)


if __name__ == "__main__":
    main()
