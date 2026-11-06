"""Self-contained tests for PhysioNet 2019 utility normalization components.

Run: python src/training/test_evaluate_utility.py
"""
from __future__ import annotations

import numpy as np

from evaluate_utility import (
    FALSE_POS, DT_EARLY, DT_LATE, DT_OPTIMAL, MAX_TP,
    MIN_FN, compute_utility, utility_components,
)


def main() -> None:
    non_septic = np.zeros(12, dtype=np.int8)
    alarm, silent, optimal = utility_components(non_septic)
    assert np.allclose(alarm, FALSE_POS)
    assert np.allclose(silent, 0)
    assert not optimal.any()
    assert compute_utility(non_septic, np.ones(12, dtype=np.int8)) == 12 * FALSE_POS

    labels = np.array([0] * 24 + [1] * 12, dtype=np.int8)
    alarm, silent, best = utility_components(labels)
    onset = 24 - DT_OPTIMAL
    # A perfect prediction at optimal time carries +1.
    assert np.isclose(alarm[onset + DT_OPTIMAL], MAX_TP)
    # Early enough predictions carry the false-positive penalty.
    assert np.isclose(alarm[onset + DT_EARLY - 1], FALSE_POS)
    # At the latest allowed time TP utility reaches zero.
    assert np.isclose(alarm[onset + DT_LATE], 0)
    # False-negative penalty at latest time reaches -2.
    assert np.isclose(silent[onset + DT_LATE], MIN_FN)
    assert np.allclose(alarm[onset + DT_LATE + 1:], 0)
    assert np.allclose(silent[:onset + DT_OPTIMAL + 1], 0)
    assert np.all(best[np.maximum(0, onset + DT_EARLY):
                       np.minimum(len(labels), onset + DT_LATE + 1)] == 1)
    assert compute_utility(labels, best) > compute_utility(
        labels, np.zeros_like(labels))

    # A patient who becomes positive too early still receives a valid timeline.
    early = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    for pred in (np.zeros(5, dtype=np.int8), np.ones(5, dtype=np.int8)):
        score = compute_utility(early, pred)
        assert np.isfinite(score)

    try:
        compute_utility(labels, np.ones(35, dtype=np.int8))
    except ValueError:
        pass
    else:
        raise AssertionError("Mismatched prediction lengths must fail")
    print("PASS: official-style utility timing, penalties, normalization components")


if __name__ == "__main__":
    main()
