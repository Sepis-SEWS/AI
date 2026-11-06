"""AI-6 PhysioNet/CinC 2019 normalized utility on locked Validation only.

Implements the official evaluate_sepsis_score.py timing/normalization:
https://github.com/physionetchallenges/evaluation-2019

IMPORTANT:
- Original patient timelines (all hourly rows) are scored.
- Models need 24 prior rows: predictions on the first 23 rows are forced to 0.
- Patients with <24 rows also have no model predictions (all zeros).
- This is a VALIDATION evaluation, NOT a final Test result.
- Thresholds are a prespecified diagnostic grid; choosing the highest on
  this same Validation cohort introduces selection bias.
- Original SepsisLabel is never shifted or relabeled.
- The 6-hour cooldown alert simulation is NOT included in the official metric.

Run: python src/training/evaluate_utility.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORES = Path("reports/ensemble_validation_scores.npz")
MANIFEST = Path("data/splits/patient_splits.json")
INDEX = Path("data/splits/patient_index.csv")
OUT = Path("reports")
THRESHOLDS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)

DT_EARLY = -12
DT_OPTIMAL = -6
DT_LATE = 3
MAX_TP = 1.0
MIN_FN = -2.0
FALSE_POS = -0.05


def utility_components(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-time utility if alarm, if silent, and official best policy.

    Reproduces official timing on zero-based row positions. The challenge
    baseline uses the supplied shifted SepsisLabel WITHOUT shifting it again.
    """
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 1 or not len(labels) or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid patient labels")
    n = len(labels)
    alarm = np.full(n, FALSE_POS, dtype=float)
    silent = np.zeros(n, dtype=float)
    ideal = np.zeros(n, dtype=np.int8)
    if not labels.any():
        return alarm, silent, ideal

    # Same as official code: t_sepsis = first(SepsisLabel==1) - dt_optimal.
    sepsis_t = int(np.flatnonzero(labels)[0]) - DT_OPTIMAL
    t = np.arange(n)
    # Official value before optimal time never falls below FP penalty.
    slope_early = MAX_TP / (DT_OPTIMAL - DT_EARLY)
    onset_early = slope_early * (t - sepsis_t - DT_EARLY)
    rise = np.maximum(onset_early, FALSE_POS)
    slope_late = -MAX_TP / (DT_LATE - DT_OPTIMAL)
    fall = slope_late * (t - sepsis_t - DT_LATE)
    m_fn = MIN_FN / (DT_LATE - DT_OPTIMAL)
    fn = m_fn * (t - sepsis_t - DT_OPTIMAL)

    before_opt = t <= sepsis_t + DT_OPTIMAL
    through_late = t <= sepsis_t + DT_LATE
    alarm = np.where(before_opt, rise,
                     np.where(through_late, fall, 0.0))
    silent = np.where(before_opt, 0.0,
                      np.where(through_late, fn, 0.0))
    ideal = ((t >= sepsis_t + DT_EARLY) & through_late).astype(np.int8)
    return alarm, silent, ideal


def compute_utility(labels: np.ndarray, predictions: np.ndarray) -> float:
    alarm, silent, _ = utility_components(labels)
    predictions = np.asarray(predictions)
    if len(predictions) != len(labels) or not np.isin(predictions, (0, 1)).all():
        raise ValueError("Invalid prediction vector")
    return float(np.where(predictions == 1, alarm, silent).sum())


def main() -> None:
    if not SCORES.is_file():
        raise FileNotFoundError(f"Missing {SCORES}; run evaluate_ensemble.py first.")
    cohort = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    val_ids = cohort["validation"]
    if len(set(cohort["train"] + val_ids + cohort["test"])) != sum(
            len(cohort[k]) for k in ("train", "validation", "test")):
        raise ValueError("Overlapping locked patient splits")
    paths = pd.read_csv(INDEX).set_index("patient_id")["path"]
    with np.load(SCORES, allow_pickle=False) as data:
        pid = data["patient_ids"]
        times = data["icul_os"].astype(float)
        observed_y = data["labels"].astype(np.int8)
        xgb = data["xgboost_scores"].astype(float)
        lstm = data["lstm_scores"].astype(float)
    if not (len(pid) == len(times) == len(observed_y) == len(xgb) == len(lstm)):
        raise ValueError("Saved score arrays differ in length")
    if set(pid) - set(val_ids):
        raise ValueError("Foreign patient present in score cache")
    if not np.isfinite(xgb).all() or not np.isfinite(lstm).all():
        raise ValueError("Non-finite prediction score")
    if np.any((xgb < 0) | (xgb > 1) | (lstm < 0) | (lstm > 1)):
        raise ValueError("Model scores outside [0, 1]")

    models = ("xgboost_176", "lstm_full_pw10", "raw_blend_lstm_0.75")
    by_threshold = {(model, cutoff): 0.0 for model in models for cutoff in THRESHOLDS}
    best_sum = 0.0
    inaction_sum = 0.0
    scored_patients = 0
    expected_cursor = 0
    short_patients = 0

    for i, patient_id in enumerate(val_ids, 1):
        raw = pd.read_csv(Path(paths.loc[patient_id]), sep="|")
        raw = raw.sort_values("ICULOS", kind="stable").reset_index(drop=True)
        labels = pd.to_numeric(raw["SepsisLabel"], errors="raise").to_numpy(dtype=np.int8)
        hours = pd.to_numeric(raw["ICULOS"], errors="raise").to_numpy(dtype=float)
        if len(hours) < 24:
            short_patients += 1
            block_size = 0
        else:
            block_size = len(hours) - 23
            scored_patients += 1
        sl = slice(expected_cursor, expected_cursor + block_size)
        if block_size and not (
                np.array_equal(pid[sl], np.asarray([patient_id] * block_size, dtype=str))
                and np.array_equal(observed_y[sl], labels[23:])
                and np.allclose(times[sl], hours[23:], atol=0.0, rtol=0.0)):
            raise ValueError(f"Prediction/label/time mismatch for {patient_id}")
        base, inactive, best_policy = utility_components(labels)
        best_sum += float(np.where(best_policy == 1, base, inactive).sum())
        inaction_sum += float(inactive.sum())
        # Mandatory first 23 timestamps have no score; define prediction=0.
        for name, probabilities in (
                ("xgboost_176", xgb[sl]),
                ("lstm_full_pw10", lstm[sl]),
                ("raw_blend_lstm_0.75", 0.25 * xgb[sl] + 0.75 * lstm[sl])):
            for threshold in THRESHOLDS:
                alarms = np.zeros(len(labels), dtype=np.int8)
                if block_size:
                    alarms[23:] = probabilities >= threshold
                by_threshold[(name, threshold)] += float(
                    np.where(alarms == 1, base, inactive).sum())
        expected_cursor += block_size
        if i % 1000 == 0:
            print(f"Utility scored Validation: {i:,}/{len(val_ids):,} patients", flush=True)

    if expected_cursor != len(pid):
        raise ValueError("Not all stored prediction rows consumed")
    denominator = best_sum - inaction_sum
    if denominator <= 0:
        raise ValueError("Official best-minus-inaction normalization is not positive")
    rows = [
        {"model": model, "score_threshold": threshold,
         "normalized_utility": float((value - inaction_sum) / denominator),
         "unnormalized_utility": float(value)}
        for (model, threshold), value in by_threshold.items()
    ]
    table = pd.DataFrame(rows).sort_values(["model", "score_threshold"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "ensemble_validation_utility_by_threshold.csv", index=False)
    report = {
        "scope": "locked_validation_only",
        "total_validation_patients": len(val_ids),
        "scored_patients": scored_patients,
        "patients_with_fewer_than_24_rows": short_patients,
        "scored_rows": len(pid),
        "first_23_hours_policy": "predict 0 (insufficient observation window)",
        "test_patients_evaluated": 0,
        "official_best_utility": best_sum,
        "official_inaction_utility": inaction_sum,
        "threshold_results": rows,
        "warning": (
            "Official Challenge 2019 utility formula/timing applied to full raw "
            "Validation patient trajectories, with zero predictions where 24-row "
            "models cannot score. This is NOT an official Challenge submission; "
            "threshold selection on the same Validation is optimistic. "
            "Utility is based on hourly binary predictions, not 6h-cooldown alerts."
        ),
        "reference": "https://github.com/physionetchallenges/evaluation-2019",
    }
    target = OUT / "ensemble_validation_utility.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"Saved: {target}; Test HELD OUT")


if __name__ == "__main__":
    main()
