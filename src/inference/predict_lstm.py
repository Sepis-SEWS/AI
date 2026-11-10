"""AI-9: read-only inference for the single frozen SEWS LSTM.

Input JSON: {"observations": [{"ICULOS": 1, "HR": 80, ...}, ...]}.
Supply all chronological records available up to the decision time (not just
the last 24): age-since-last-measurement channels depend on prior history.
No SepsisLabel, patient identifier, or Test data is needed.

Run from repository root:
    python src/inference/predict_lstm.py --input-json path/to/patient.json

The output is an *uncalibrated* risk score, NOT a probability of sepsis.
This module never trains, recalibrates, rewrites artifacts, or accesses Test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "preprocessing"))
sys.path.insert(0, str(ROOT / "src" / "training"))
from lstm_sequences import FEATURES, WINDOW, make_sequences  # noqa: E402
from train_lstm import RiskLSTM  # noqa: E402

MODEL_FILE = "models/lstm_full_pw10_best.pt"
STATS_FILE = "models/lstm_preprocessing_stats.json"
POLICY_FILE = "reports/final_evaluation_policy_locked.json"
THRESHOLD = 0.2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_hash(policy: dict[str, Any], relative_path: str) -> str:
    matches = [value for key, value in policy.get("artifact_sha256", {}).items()
               if key.replace("\\", "/") == relative_path]
    if len(matches) != 1:
        raise ValueError(f"Missing or ambiguous locked checksum: {relative_path}")
    return matches[0]


def load_frozen_model(root: Path = ROOT) -> tuple[RiskLSTM, dict[str, Any]]:
    """Check frozen artifacts *before* loading; never modify them."""
    policy = json.loads((root / POLICY_FILE).read_text(encoding="utf-8"))
    expected = {
        "status": "LOCKED_BEFORE_TEST", "model": "lstm_full_pw10",
        "score_threshold": THRESHOLD,
        "score_type": "uncalibrated_sigmoid_logit_output",
        "use_sigmoid_calibrator": False, "window_rows": WINDOW,
        "channels": len(FEATURES) * 4,
        "first_23_rows": "predict_zero",
        "patients_fewer_than_24_rows": "predict_zero_at_all_rows",
        "alert_policy": "hourly_binary_predictions_no_cooldown",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"Frozen policy mismatch: {key}")
    for key, file in (("checkpoint", MODEL_FILE), ("preprocessing_stats", STATS_FILE)):
        if str(policy.get(key, "")).replace("\\", "/") != file:
            raise ValueError(f"Frozen policy path mismatch: {key}")
        actual = file_sha256(root / file)
        if actual != _policy_hash(policy, file):
            raise ValueError(f"Frozen file checksum mismatch: {file}")

    stats = json.loads((root / STATS_FILE).read_text(encoding="utf-8"))
    if (stats.get("features") != FEATURES or stats.get("window_rows") != WINDOW
            or stats.get("channels_per_feature") !=
            ["scaled_value", "original_mask", "age_hours_scaled", "never_observed"]
            or stats.get("train_patients_used") != 25814
            or stats.get("train_patients_in_manifest") != 25814):
        raise ValueError("Frozen preprocessing schema/fit scope mismatch")
    checkpoint = torch.load(root / MODEL_FILE, map_location="cpu", weights_only=True)
    if (checkpoint.get("scope") != "full"
            or checkpoint.get("run_tag") != ""
            or checkpoint.get("features") != FEATURES
            or checkpoint.get("input_shape") != [WINDOW, len(FEATURES) * 4]
            or checkpoint.get("best_epoch") != 6
            or checkpoint.get("pos_weight") != 10
            or checkpoint.get("hidden_size") != 64
            or checkpoint.get("layers") != 1):
        raise ValueError("Unexpected frozen LSTM checkpoint metadata")
    model = RiskLSTM(hidden=64, layers=1)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, stats


def build_inference_frame(observations: list[dict[str, Any]]) -> pd.DataFrame:
    """Validate chronological unlabeled observations; no synthetic clinical data."""
    if not isinstance(observations, list) or not observations:
        raise ValueError("observations must be a nonempty list of row objects")
    if not all(isinstance(row, dict) for row in observations):
        raise ValueError("Every observation must be a JSON object")
    if any("SepsisLabel" in row for row in observations):
        raise ValueError("Inference input must not contain SepsisLabel")
    frame = pd.DataFrame(observations)
    required = ["ICULOS", *FEATURES]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    frame = frame.loc[:, required].copy()
    hours = pd.to_numeric(frame["ICULOS"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(hours).all() or np.any(np.diff(hours) <= 0):
        raise ValueError("ICULOS must be finite, strictly increasing, without duplicates")
    frame["ICULOS"] = hours
    for name in FEATURES:
        # Missing observations may be null/NaN; other nonnumeric values are rejected.
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"Infinite observation in {name}")
        frame[name] = values
    # Training preprocessing requires SepsisLabel even though inference is unlabeled.
    # All-zero synthetic labels are discarded; only x[-1] is used.
    frame["SepsisLabel"] = 0
    return frame


def predict(observations: list[dict[str, Any]], model: RiskLSTM,
            stats: dict[str, Any]) -> dict[str, Any]:
    frame = build_inference_frame(observations)
    if len(frame) < WINDOW:
        return {
            "status": "insufficient_history",
            "risk_score": None,
            "alert": False,
            "score_threshold": THRESHOLD,
            "records_received": len(frame),
            "minimum_records": WINDOW,
            "calibrated_probability": None,
        }
    x, _ = make_sequences(frame, stats)
    if x.shape[1:] != (WINDOW, len(FEATURES) * 4) or not np.isfinite(x[-1]).all():
        raise ValueError("Preprocessed inference tensor incompatible with frozen LSTM")
    with torch.inference_mode():
        logits = model(torch.from_numpy(x[-1:]))
        score = float(torch.sigmoid(logits).item())
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid LSTM score")
    return {
        "status": "scored",
        "risk_score": score,
        "alert": bool(score >= THRESHOLD),
        "score_threshold": THRESHOLD,
        "records_received": len(frame),
        "minimum_records": WINDOW,
        "calibrated_probability": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True,
                        help="A JSON list of full patient history rows, or an observations object")
    args = parser.parse_args()
    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    observations = data.get("observations") if isinstance(data, dict) else data
    model, stats = load_frozen_model()
    result = predict(observations, model, stats)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
