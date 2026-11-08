"""Independent reference parity test for 2019 challenge utility.

Reference: https://github.com/physionetchallenges/evaluation-2019
             blob/master/evaluate_sepsis_score.py
Tests the local vectorized patient utility against a separate, scalar implementation
of the official scoring branches, including septic and non-septic patients,
very early onset, very late onset, and randomized binary predictions.

Usage: python src/training/test_utility_official_parity.py
No patient files or Test data are accessed.
"""
from __future__ import annotations

import numpy as np

from evaluate_utility import compute_utility, utility_components


def official_scalar_reference(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Independent scalar port of official compute_prediction_utility."""
    dt_early, dt_optimal, dt_late = -12, -6, 3
    max_u_tp, min_u_fn, u_fp, u_tn = 1.0, -2.0, -0.05, 0.0
    is_septic = bool(np.any(labels))
    t_sepsis = int(np.argmax(labels)) - dt_optimal if is_septic else float("inf")
    m_1 = max_u_tp / (dt_optimal - dt_early)
    b_1 = -m_1 * dt_early
    m_2 = -max_u_tp / (dt_late - dt_optimal)
    b_2 = -m_2 * dt_late
    m_3 = min_u_fn / (dt_late - dt_optimal)
    b_3 = -m_3 * dt_optimal
    result = 0.0
    for t in range(len(labels)):
        u = 0.0
        if t <= t_sepsis + dt_late:
            if is_septic and predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u = max(m_1 * (t - t_sepsis) + b_1, u_fp)
                else:
                    u = m_2 * (t - t_sepsis) + b_2
            elif not is_septic and predictions[t]:
                u = u_fp
            elif is_septic and not predictions[t]:
                if t <= t_sepsis + dt_optimal:
                    u = 0.0
                else:
                    u = m_3 * (t - t_sepsis) + b_3
            elif not is_septic and not predictions[t]:
                u = u_tn
        result += u
    return result


def official_best(labels: np.ndarray) -> np.ndarray:
    prediction = np.zeros(len(labels), dtype=np.int8)
    if labels.any():
        t_sepsis = int(np.argmax(labels)) + 6
        prediction[max(0, t_sepsis - 12):min(t_sepsis + 4, len(labels))] = 1
    return prediction


def check(labels: np.ndarray, predictions: np.ndarray) -> None:
    expected = official_scalar_reference(labels, predictions)
    actual = compute_utility(labels, predictions)
    if not np.isclose(actual, expected, rtol=1e-12, atol=1e-10):
        raise AssertionError(f"Official parity failed: {actual} vs {expected}")
    alarm, silent, best = utility_components(labels)
    for selected in (np.zeros_like(labels), np.ones_like(labels),
                     predictions, official_best(labels)):
        expected = official_scalar_reference(labels, selected)
        actual = float(np.where(selected, alarm, silent).sum())
        if not np.isclose(actual, expected, rtol=1e-12, atol=1e-10):
            raise AssertionError(f"Branch parity failed: {actual} vs {expected}")
    if not np.array_equal(best, official_best(labels)):
        raise AssertionError("Best-policy vector differs from official reference")


def main() -> None:
    rng = np.random.default_rng(42)
    checked = 0
    for length in (1, 2, 5, 12, 23, 24, 36, 48, 96, 200):
        for onset in (None, 0, 1, length // 2, length - 1):
            labels = np.zeros(length, dtype=np.int8)
            if onset is not None:
                labels[onset:] = 1
            for _ in range(25):
                predictions = rng.integers(0, 2, size=length, dtype=np.int8)
                check(labels, predictions)
                checked += 1
    print(f"PASS: {checked} deterministic parity cases vs independent official scoring branches")
    print("No data accessed. Test HELD OUT.")


if __name__ == "__main__":
    main()
