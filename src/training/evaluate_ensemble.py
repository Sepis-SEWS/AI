"""AI-6: aligned XGBoost/LSTM validation-score ensemble (NO retraining).

Usage:
    python src/training/evaluate_ensemble.py

Uses locked Validation patients only, same 24-row end-of-window labels.
Saves row-level aligned scores for later Utility Score/calibration analysis.
Raw-score blends are diagnostic: both models use weighted losses and are not
calibrated probabilities. Selecting an ensemble weight on the same Validation
set has selection bias. Test stays entirely held out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_xgboost import CACHE_DIR, INDEX, MANIFEST, load_or_build
from experiment_missingness import EXTRA_NAMES, chunks_for
from train_xgboost import FEATURES, VARS
from train_lstm import RiskLSTM

PREPROCESS_DIR = Path(__file__).resolve().parent.parent / "preprocessing"
sys.path.insert(0, str(PREPROCESS_DIR))
from lstm_sequences import make_sequences, patient_path, read_manifest  # noqa: E402

XGB_MODEL = Path("models/xgboost_ablation_full_age_only.joblib")
LSTM_MODEL = Path("models/lstm_full_pw10_best.pt")
STATS = Path("models/lstm_preprocessing_stats.json")
OUT = Path("reports")


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    return {
        "AUROC": float(roc_auc_score(labels, probabilities)),
        "AP": float(average_precision_score(labels, probabilities)),
        "positive_rate": float(labels.mean()),
    }


def main() -> None:
    groups = read_manifest(MANIFEST)
    ids = groups["validation"]
    index = pd.read_csv(INDEX)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate IDs in index")
    index = index.set_index("patient_id")
    if not set(ids).issubset(index.index):
        raise ValueError("Validation patient missing from index")

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if (stats.get("train_patients_used") != len(groups["train"]) or
            stats.get("train_patients_in_manifest") != len(groups["train"])):
        raise ValueError("LSTM preprocessing stats not fitted on full Train cohort")
    paths = [Path(index.loc[pid, "path"]) for pid in ids]
    base, y, patient_ids = load_or_build("validation", ids, paths, CACHE_DIR, False)
    extra = chunks_for("validation", ids, paths, CACHE_DIR, False)
    if len(base) != len(y) or len(extra) != len(y) or len(patient_ids) != len(y):
        raise ValueError("Unaligned XGBoost caches")
    if not set(patient_ids).issubset(set(ids)):
        raise ValueError("Foreign patient in Validation cache")
    age_names = [f"{var}_age_hours" for var in VARS]
    features = FEATURES + age_names
    xgb_saved = joblib.load(XGB_MODEL)
    if xgb_saved.get("features") != features:
        raise ValueError("Unexpected XGBoost feature order")
    age_idx = [EXTRA_NAMES.index(n) for n in age_names]
    frame = pd.DataFrame(np.column_stack((base, extra[:, age_idx])), columns=features)
    xgb_score = xgb_saved["model"].predict_proba(frame)[:, 1]

    # PyTorch weights-only loading avoids arbitrary pickle execution from checkpoint.
    checkpoint = torch.load(LSTM_MODEL, map_location="cpu", weights_only=True)
    if checkpoint.get("scope") != "full" or checkpoint.get("input_shape") != [24, 64]:
        raise ValueError("LSTM checkpoint does not match full 24x64 experiment")
    if checkpoint.get("features") != stats["features"]:
        raise ValueError("LSTM checkpoint and preprocessing schema disagree")
    model = RiskLSTM(hidden=int(checkpoint["hidden_size"]), layers=int(checkpoint["layers"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Validation patients are processed in the exact same patient order as cache.
    predicted = []
    matching_labels = []
    matching_ids = []
    matching_times = []
    with torch.inference_mode():
        for i, pid in enumerate(ids, 1):
            df = pd.read_csv(patient_path(Path("data/data-sepsis/training"), pid), sep="|")
            x, local_y = make_sequences(df, stats)
            if not len(local_y):
                continue
            chunks = []
            for start in range(0, len(local_y), 256):
                batch = torch.from_numpy(x[start:start + 256])
                chunks.append(torch.sigmoid(model(batch)).numpy())
            predicted.append(np.concatenate(chunks))
            matching_labels.append(local_y)
            matching_ids.extend([pid] * len(local_y))
            times = pd.to_numeric(df["ICULOS"], errors="raise").sort_values(
                kind="stable").to_numpy()
            matching_times.extend(times[23:].tolist())
            if i % 500 == 0:
                print(f"Scored LSTM Validation patients: {i:,}/{len(ids):,}", flush=True)

    lstm_score = np.concatenate(predicted)
    labels_lstm = np.concatenate(matching_labels)
    if (len(lstm_score) != len(xgb_score) or
            not np.array_equal(labels_lstm, y) or
            not np.array_equal(np.asarray(matching_ids, dtype=str), patient_ids)):
        raise ValueError("XGBoost and LSTM predictions NOT aligned; no ensemble scored")

    results = [
        {"model": "xgboost_176", **metrics(y, xgb_score)},
        {"model": "lstm_full_pw10", **metrics(y, lstm_score)},
    ]
    for weight in (0.25, 0.5, 0.75):
        pred = (1 - weight) * xgb_score + weight * lstm_score
        results.append({"model": f"raw_blend_lstm_{weight:g}", **metrics(y, pred)})
    # Ranks remove scale mismatch but use complete Validation score distribution;
    # do not mistake this for prospective inference or calibrated probabilities.
    xgb_rank = rankdata(xgb_score, method="average") / len(y)
    lstm_rank = rankdata(lstm_score, method="average") / len(y)
    for weight in (0.25, 0.5, 0.75):
        pred = (1 - weight) * xgb_rank + weight * lstm_rank
        results.append({"model": f"rank_blend_lstm_{weight:g}", **metrics(y, pred)})
    table = pd.DataFrame(results)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ensemble_validation_comparison.csv", index=False)
    np.savez_compressed(OUT / "ensemble_validation_scores.npz",
                        patient_ids=np.asarray(matching_ids, dtype=str),
                        icul_os=np.asarray(matching_times, dtype=np.float32),
                        labels=y, xgboost_scores=xgb_score,
                        lstm_scores=lstm_score)
    report = {
        "scope": "full_locked_validation_only",
        "validation_patients": len(ids), "validation_rows": len(y),
        "test_patients_evaluated": 0,
        "lstm_checkpoint_epoch": checkpoint["best_epoch"],
        "results": results,
        "warning": ("All blends are diagnostic; no calibration or clinical threshold. "
                    "Rank blends rely on entire validation cohort and are not "
                    "deployable as-is. Weight selection on the same Validation is "
                    "optimistically biased. Do not score Test until final lock."),
    }
    (OUT / "ensemble_validation_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print("Saved Validation comparison/row-level scores; Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
