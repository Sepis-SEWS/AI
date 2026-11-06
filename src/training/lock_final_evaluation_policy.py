"""Lock SEWS final evaluation policy before ever scoring Test patients.

Run: python src/training/lock_final_evaluation_policy.py

This step accesses NO raw Test patient data. It checks the preflight hashes
against current model artifacts, then freezes a SINGLE primary candidate:
LSTM full, pos_weight=10, 24-row causal window, raw-score threshold 0.2,
no sigmoid calibration, first 23 rows predict no alert, no cooldown.
Threshold was selected on the reused Validation set and is NOT clinical.
The secondary 25:75 ensemble is documented but NOT selected for final Test.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FREEZE = Path("reports/model_pretest_freeze.json")
UTILITY = Path("reports/ensemble_validation_utility.json")
OUT = Path("reports/final_evaluation_policy_locked.json")
THRESHOLD = 0.2


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Final policy already locked: {OUT}; do not overwrite")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))
    if frozen["test_patients_evaluated"] != 0:
        raise ValueError("Preflight report is not Test held-out")
    if frozen["patients"] != {
        "train": 25814, "validation": 6454, "test_held_out": 8068,
        "validation_patients_with_24row_predictions": 4960,
        "validation_scored_rows": 108748,
    }:
        raise ValueError("Unexpected cohort counts in preflight")
    if frozen["primary_candidate"]["model"] != "lstm_full_pw10":
        raise ValueError("Unexpected primary candidate")
    expected_hashes = frozen["artifacts_sha256"]
    for name, expected in expected_hashes.items():
        target = Path(name)
        if not target.is_file() or checksum(target) != expected:
            raise ValueError(f"Artifact was modified since preflight: {name}")
    utility = json.loads(UTILITY.read_text(encoding="utf-8"))
    exact = [x for x in utility["threshold_results"] if
             x["model"] == "lstm_full_pw10" and
             abs(float(x["score_threshold"]) - THRESHOLD) < 1e-12]
    if len(exact) != 1:
        raise ValueError("Primary candidate threshold not evaluated on Validation")
    report = {
        "status": "LOCKED_BEFORE_TEST",
        "model": "lstm_full_pw10",
        "score_threshold": THRESHOLD,
        "score_type": "uncalibrated_sigmoid_logit_output",
        "use_sigmoid_calibrator": False,
        "window_rows": 24,
        "channels": 64,
        "first_23_rows": "predict_zero",
        "patients_fewer_than_24_rows": "predict_zero_at_all_rows",
        "alert_policy": "hourly_binary_predictions_no_cooldown",
        "utility": "PhysioNet Challenge 2019 per-patient normalized utility",
        "checkpoint": "models/lstm_full_pw10_best.pt",
        "preprocessing_stats": "models/lstm_preprocessing_stats.json",
        "patient_split": "data/splits/patient_splits.json",
        "artifact_sha256": expected_hashes,
        "preflight_report_sha256": checksum(FREEZE),
        "validation_utility_at_selected_threshold": float(exact[0]["normalized_utility"]),
        "validation_based_choice_warning": (
            "Chosen after repeated fixed Validation analysis; biased development "
            "estimate. Test must be scored once without reselecting threshold."
        ),
        "secondary_ensemble_policy": "recorded for research, NOT a final Test candidate",
        "test_evaluated": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"LOCKED: {report['model']} threshold={THRESHOLD}")
    print(f"SHA256-verified {len(expected_hashes)} preflight artifacts")
    print(f"Saved: {OUT}. Test HELD OUT.")


if __name__ == "__main__":
    main()
