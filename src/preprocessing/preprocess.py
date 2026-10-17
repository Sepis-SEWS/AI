"""Causal per-patient preprocessing for SEWS (AI-3).

The limits below are INITIAL EXPERIMENT SETTINGS, not clinically validated
measurement-validity periods. Laboratory values remain missing by default.
No train-set statistics are estimated in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total",
    "Platelets", "FiO2", "BUN", "Glucose", "pH",
]

# Candidate ages (elapsed ICULOS hours). None = keep missing, no fill.
FFILL_MAX_AGE_HOURS: dict[str, float | None] = {
    "HR": 2, "Temp": 6, "SBP": 2, "DBP": 2, "MAP": 2,
    "Resp": 2, "O2Sat": 2,
    "WBC": None, "Lactate": None, "Creatinine": None,
    "Bilirubin_total": None, "Platelets": None, "FiO2": None,
    "BUN": None, "Glucose": None, "pH": None,
}


def preprocess_patient(
    frame: pd.DataFrame,
    max_age_hours: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    """Process ONE patient's chronological observations without future leakage.

    Columns per feature:
      feature_raw: original numeric observation, NaN preserved
      feature_mask: 1 only if measured at the current row
      feature_age_hours: time since last real measurement (NaN if never)
      feature_count_24: number of real observations in current + previous 23 rows
      feature: optional age-limited forward-fill (laboratories default to NaN)
    The target SepsisLabel is carried through unchanged.
    """
    required = set(FEATURES) | {"ICULOS", "SepsisLabel"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Patient has no rows")

    policy = dict(FFILL_MAX_AGE_HOURS)
    if max_age_hours is not None:
        unknown = set(max_age_hours).difference(FEATURES)
        if unknown:
            raise ValueError(f"Unknown features in policy: {sorted(unknown)}")
        policy.update(max_age_hours)
    if any(age is not None and (not np.isfinite(age) or age < 0)
           for age in policy.values()):
        raise ValueError("All forward-fill maximum ages must be nonnegative and finite")

    out = frame.copy()
    out["ICULOS"] = pd.to_numeric(out["ICULOS"], errors="coerce")
    times = out["ICULOS"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or len(np.unique(times)) != len(times):
        raise ValueError("Invalid or duplicate ICULOS")
    out = out.sort_values("ICULOS", kind="stable").reset_index(drop=True)
    times = out["ICULOS"]

    for feature in FEATURES:
        raw = pd.to_numeric(out[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        measured = raw.notna()
        out[f"{feature}_raw"] = raw
        out[f"{feature}_mask"] = measured.astype("int8")
        out[f"{feature}_count_24"] = (
            measured.astype("int32").rolling(24, min_periods=1).sum().astype("int16")
        )
        last_measured_time = times.where(measured).ffill()
        age = times - last_measured_time
        out[f"{feature}_age_hours"] = age.astype(float)
        threshold = policy[feature]
        if threshold is None:
            out[feature] = raw
        else:
            candidate = raw.ffill()
            out[feature] = candidate.where(age.le(threshold))

    return out
