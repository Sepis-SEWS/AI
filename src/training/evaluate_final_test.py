"""SEWS final one-time locked Test evaluation.

Run from repository root after lock_final_evaluation_policy.py:
    python src/training/evaluate_final_test.py

HARD RULES:
- Verify the locked policy and all preflight SHA256 artifact hashes.
- Only one previously locked model (LSTM full, raw threshold=0.2).
- No Test-driven tuning, retraining, calibration, or threshold search.
- Include all Test patients in official-style utility, predicting 0 until
  24 completed rows; window-based AUROC/AP use only scoreable rows.
- Refuse to overwrite a previous result. Keep patient predictions in memory.
- No patient-level or raw clinical records exported.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))

from lstm_sequences import FEATURES, make_sequences, patient_path, read_manifest  # noqa: E402
from train_lstm import RiskLSTM  # noqa: E402
from evaluate_utility import utility_components  # noqa: E402

POLICY = Path("reports/final_evaluation_policy_locked.json")
RESULT = Path("reports/final_test_evaluation.json")
MODEL = Path("models/lstm_full_pw10_best.pt")
STATS = Path("models/lstm_preprocessing_stats.json")
ROOT = Path("data/data-sepsis/training")
THRESHOLD = 0.2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    b = np.searchsorted(np.linspace(0, 1, bins + 1)[1:-1], p, side="right")
    return float(sum(np.mean(b == i) * abs(np.mean(y[b == i]) - np.mean(p[b == i]))
                     for i in range(bins) if np.any(b == i)))


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(f"Refusing to repeat/overwrite locked Test evaluation: {RESULT}")
    if not POLICY.is_file():
        raise FileNotFoundError("Missing locked policy; run lock_final_evaluation_policy.py")
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    if policy.get("status") != "LOCKED_BEFORE_TEST" or policy.get("test_evaluated") is not False:
        raise ValueError("Policy is not a pre-Test lock")
    required = {
        "model": "lstm_full_pw10", "score_threshold": THRESHOLD,
        "score_type": "uncalibrated_sigmoid_logit_output",
        "use_sigmoid_calibrator": False,
        "window_rows": 24, "channels": 64,
        "first_23_rows": "predict_zero",
        "patients_fewer_than_24_rows": "predict_zero_at_all_rows",
        "alert_policy": "hourly_binary_predictions_no_cooldown",
        "checkpoint": str(MODEL),
        "preprocessing_stats": str(STATS),
        "patient_split": "data/splits/patient_splits.json",
    }
    for key, expected in required.items():
        if policy.get(key) != expected:
            raise ValueError(f"Locked policy mismatch on {key}")
    if sha256_file(Path("reports/model_pretest_freeze.json")) != policy["preflight_report_sha256"]:
        raise ValueError("Preflight freeze file changed after policy lock")
    for file, expected_hash in policy["artifact_sha256"].items():
        if sha256_file(Path(file)) != expected_hash:
            raise ValueError(f"Preflight artifact hash changed: {file}")

    groups = read_manifest(Path(policy["patient_split"]))
    if [len(groups[k]) for k in ("train", "validation", "test")] != [25814, 6454, 8068]:
        raise ValueError("Locked cohort has unexpected sizes")
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if (stats.get("train_patients_used") != len(groups["train"]) or
            stats.get("train_patients_in_manifest") != len(groups["train"]) or
            stats["window_rows"] != 24 or stats["features"] != FEATURES):
        raise ValueError("Train-only preprocessing stats incompatible")
    ckpt = torch.load(MODEL, map_location="cpu", weights_only=True)
    if (ckpt.get("scope") != "full" or ckpt.get("input_shape") != [24, 64] or
            ckpt.get("features") != FEATURES or ckpt.get("pos_weight") != 10 or
            ckpt.get("best_epoch") != 6):
        raise ValueError("Unexpected locked LSTM checkpoint")
    model = RiskLSTM(hidden=int(ckpt["hidden_size"]), layers=int(ckpt["layers"]))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    all_labels, all_scores = [], []
    total_utility = best_utility = inaction_utility = 0.0
    total_rows = scoreable_rows = short_patients = 0
    tp = fp = fn = tn = 0
    print("ONE-TIME FINAL Test scoring: LOCKED LSTM, threshold=0.2; no changes permitted", flush=True)
    with torch.inference_mode():
        for i, pid in enumerate(groups["test"], 1):
            frame = pd.read_csv(patient_path(ROOT, pid), sep="|")
            x, y = make_sequences(frame, stats)
            # Raw input chronology is restored to match preprocess_patient sorting.
            frame = frame.sort_values("ICULOS", kind="stable").reset_index(drop=True)
            full_labels = pd.to_numeric(frame["SepsisLabel"], errors="raise").to_numpy(dtype=np.int8)
            if len(x) != max(len(full_labels) - 23, 0) or not np.array_equal(y, full_labels[23:]):
                raise ValueError(f"Time/label misalignment for {pid}")
            all_predictions = np.zeros(len(full_labels), dtype=np.int8)
            if len(y):
                scores = []
                for start in range(0, len(y), 256):
                    batch = torch.from_numpy(x[start:start + 256])
                    scores.append(torch.sigmoid(model(batch)).numpy())
                p = np.concatenate(scores)
                if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                    raise ValueError(f"Invalid model scores for {pid}")
                all_scores.append(p)
                all_labels.append(y)
                all_predictions[23:] = (p >= THRESHOLD).astype(np.int8)
                scoreable_rows += len(y)
            else:
                short_patients += 1
            total_rows += len(full_labels)
            alarm, silent, ideal = utility_components(full_labels)
            total_utility += float(np.where(all_predictions == 1, alarm, silent).sum())
            best_utility += float(np.where(ideal == 1, alarm, silent).sum())
            inaction_utility += float(silent.sum())
            tp += int(np.sum((all_predictions == 1) & (full_labels == 1)))
            fp += int(np.sum((all_predictions == 1) & (full_labels == 0)))
            fn += int(np.sum((all_predictions == 0) & (full_labels == 1)))
            tn += int(np.sum((all_predictions == 0) & (full_labels == 0)))
            if i % 500 == 0:
                print(f"Test patients: {i:,}/{len(groups['test']):,}", flush=True)

    y_all = np.concatenate(all_labels)
    p_all = np.concatenate(all_scores)
    if len(y_all) != scoreable_rows or len(np.unique(y_all)) != 2:
        raise ValueError("Final AUROC/AP undefined or scored-row count mismatch")
    denominator = best_utility - inaction_utility
    if denominator <= 0:
        raise ValueError("Invalid official utility normalization denominator")
    report = {
        "status": "FINAL_LOCKED_TEST_EVALUATION_COMPLETE",
        "test_patients": len(groups["test"]), "test_total_hourly_rows": total_rows,
        "test_scored_patients": len(groups["test"]) - short_patients,
        "test_patients_shorter_than_24_rows": short_patients,
        "test_scoreable_rows": scoreable_rows,
        "model": policy["model"], "raw_score_threshold": THRESHOLD,
        "test_scoreable_row_positive_rate": float(y_all.mean()),
        "test_scoreable_row_AUROC": float(roc_auc_score(y_all, p_all)),
        "test_scoreable_row_AP": float(average_precision_score(y_all, p_all)),
        "test_scoreable_row_brier": float(brier_score_loss(y_all, p_all)),
        "test_scoreable_row_ECE_10_bins": ece(y_all, p_all),
        "full_hourly_threshold_confusion_matrix": {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
        },
        "full_patient_timeline_normalized_utility":
            float((total_utility - inaction_utility) / denominator),
        "full_patient_timeline_raw_utility": total_utility,
        "best_utility": best_utility, "inaction_utility": inaction_utility,
        "policy_file_sha256": sha256_file(POLICY),
        "limitations": (
            "One-time evaluation of a previously locked model and threshold, "
            "not a clinical deployment claim. Test was not used for tuning. "
            "AUROC/AP/calibration apply only from completed 24-row windows. "
            "Utility and hourly confusion matrix include all Test patient hours "
            "with forced zero predictions before the first 24-row window. "
            "No cooldown and no sigmoid recalibration."
        ),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(json.dumps({
        k: report[k] for k in (
            "test_patients", "test_scored_patients", "test_scoreable_rows",
            "test_scoreable_row_AUROC", "test_scoreable_row_AP",
            "test_scoreable_row_brier", "test_scoreable_row_ECE_10_bins",
            "full_patient_timeline_normalized_utility")
    }, indent=2), flush=True)
    print(f"Saved one-time final Test report: {RESULT}", flush=True)


if __name__ == "__main__":
    main()
