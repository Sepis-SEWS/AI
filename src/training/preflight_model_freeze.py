"""SEWS pre-Test model freeze/preflight: record artifacts without evaluating Test.

Run from repository root:
  python src/training/preflight_model_freeze.py

This audit checks locked patient membership and existence of local artifacts,
writes SHA256 manifest for reproducibility, and records candidate policies.
It neither reads raw patient files nor loads/scorers Test patients.
NO Test evaluation runs from this command.

The 0.20 score threshold is selected on an already reused Validation cohort,
so reported utility is a provisional development result, not final Test score.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(".")
SPLIT = ROOT / "data/splits/patient_splits.json"
STATS = ROOT / "models/lstm_preprocessing_stats.json"
XGBOOST = ROOT / "models/xgboost_ablation_full_age_only.joblib"
LSTM = ROOT / "models/lstm_full_pw10_best.pt"
PREDICTIONS = ROOT / "reports/ensemble_validation_scores.npz"
UTILITY = ROOT / "reports/ensemble_validation_utility.json"
CALIBRATION = ROOT / "reports/validation_patient_split_calibration.json"
SHAP = ROOT / "reports/xgboost_age176_shap_summary.json"
OUTPUT = ROOT / "reports/model_pretest_freeze.json"
REQUIRED = (SPLIT, STATS, XGBOOST, LSTM, PREDICTIONS, UTILITY, CALIBRATION, SHAP)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> None:
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model-freeze prerequisites: {missing}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["split"]
    train, validation, test = (split[k] for k in ("train", "validation", "test"))
    if [len(train), len(validation), len(test)] != [25814, 6454, 8068]:
        raise ValueError("Locked patient split count mismatch")
    if len(set(train + validation + test)) != len(train + validation + test):
        raise ValueError("Patient leakage between locked splits")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if stats.get("train_patients_used") != len(train):
        raise ValueError("LSTM stats were not fitted on full locked Train cohort")
    xgb = joblib.load(XGBOOST)
    if len(xgb.get("features", [])) != 176:
        raise ValueError("Unexpected XGBoost features")
    checkpoint = torch.load(LSTM, map_location="cpu", weights_only=True)
    if checkpoint.get("scope") != "full":
        raise ValueError("LSTM checkpoint is not the full experiment")
    if checkpoint.get("input_shape") != [24, 64]:
        raise ValueError("Unexpected LSTM shape")
    if checkpoint.get("pos_weight") != 10:
        raise ValueError("Unexpected LSTM loss weighting")
    if checkpoint.get("features") != stats.get("features"):
        raise ValueError("LSTM checkpoint/statistics feature mismatch")
    with np.load(PREDICTIONS, allow_pickle=False) as predicted:
        ids = predicted["patient_ids"]
        label = predicted["labels"]
        xgb_score = predicted["xgboost_scores"]
        lstm_score = predicted["lstm_scores"]
    if not (len(ids) == len(label) == len(xgb_score) == len(lstm_score)):
        raise ValueError("Misaligned saved validation scores")
    if not set(ids).issubset(set(validation)):
        raise ValueError("Unexpected patient in Validation scoring artifact")
    if not np.isin(label, [0, 1]).all():
        raise ValueError("Validation label is not binary")
    if not (np.isfinite(xgb_score).all() and np.isfinite(lstm_score).all()):
        raise ValueError("Nonfinite validation predictions")
    utility = json.loads(UTILITY.read_text(encoding="utf-8"))
    if (utility.get("total_validation_patients") != len(validation) or
            utility.get("test_patients_evaluated") != 0):
        raise ValueError("Utility report does not match locked Validation-only run")
    if not any(
            r["model"] == "raw_blend_lstm_0.75" and
            abs(float(r["score_threshold"]) - 0.2) < 1e-9
            for r in utility["threshold_results"]):
        raise ValueError("Planned ensemble/threshold was not evaluated on Validation")
    report = {
        "scope": "pretest_validation_freeze_only",
        "status": "provisional_policy_not_clinical",
        "patients": {
            "train": len(train), "validation": len(validation),
            "test_held_out": len(test),
            "validation_patients_with_24row_predictions": len(set(ids)),
            "validation_scored_rows": len(label),
        },
        "primary_candidate": {
            "model": "lstm_full_pw10",
            "checkpoint": str(LSTM),
            "selection_reason": "simpler and better calibration before correction",
        },
        "secondary_candidate": {
            "model": "raw_blend_lstm_0.75",
            "composition": "0.25*xgboost176+0.75*lstm_full_pw10",
            "exploratory_utility_threshold": 0.2,
            "selection_reason": "higher Validation AUROC/AP/utility; "
                                "bootstrap CI included zero for lift vs LSTM",
        },
        "preprocessing": {
            "window_rows": 24,
            "lstm_channels": 64,
            "first_23_hours_policy": "no alert (no model window)",
            "calibration_status": "diagnostic sigmoid study only; no final calibrator",
        },
        "artifacts_sha256": {str(path): digest(path) for path in REQUIRED},
        "test_patients_evaluated": 0,
        "limitations": (
            "Repeated use of fixed Validation for model/weight/threshold choice "
            "induces selection bias. This manifests files, not independent "
            "performance. Before any final Test evaluation, separately verify "
            "official utility parity and freeze one primary model plus a "
            "consistent prediction threshold and scoring policy. "
            "Do not silently turn this provisional report into clinical policy."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "model": report["primary_candidate"]["model"],
        "alternative": report["secondary_candidate"]["model"],
        "counts": report["patients"],
        "artifacts_verified": len(REQUIRED),
        "test_patients_evaluated": 0,
        "saved": str(OUTPUT),
    }, indent=2))
    print("Test HELD OUT. No model training or patient scoring performed.")


if __name__ == "__main__":
    main()
