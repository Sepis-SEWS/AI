"""Unit tests for SEWS causal preprocessing; run with Python directly."""
import numpy as np
import pandas as pd

from preprocess import FEATURES, preprocess_patient


def make_patient() -> pd.DataFrame:
    data = pd.DataFrame({
        "ICULOS": [1, 2, 4, 5, 10],
        "SepsisLabel": [0, 0, 0, 1, 1],
        **{name: [np.nan] * 5 for name in FEATURES},
    })
    data["HR"] = [85.0, np.nan, np.nan, np.nan, 105.0]
    data["Temp"] = [np.nan, 36.5, np.nan, np.nan, np.nan]
    data["Lactate"] = [2.5, np.nan, np.nan, np.nan, np.nan]
    return data


def main() -> None:
    result = preprocess_patient(make_patient())
    # HR uses last real observation only within 2 elapsed ICULOS hours.
    assert result["HR"].iloc[0] == 85
    assert result["HR"].iloc[1] == 85
    assert pd.isna(result["HR"].iloc[2])  # ICULOS=4, age=3
    assert pd.isna(result["HR"].iloc[3])
    assert result["HR"].iloc[4] == 105
    assert result["HR_mask"].tolist() == [1, 0, 0, 0, 1]
    assert result["HR_count_24"].tolist() == [1, 1, 1, 1, 2]
    assert result["HR_age_hours"].tolist() == [0, 1, 3, 4, 0]
    # Temperature: no backward filling; elapsed hours, not row count.
    assert pd.isna(result["Temp"].iloc[0])
    assert result["Temp"].iloc[3] == 36.5
    assert pd.isna(result["Temp"].iloc[4])  # 8 elapsed hours from ICULOS=2
    # Sparse lab stays missing even though a previous result exists.
    assert pd.isna(result["Lactate"].iloc[1])
    assert result["Lactate_raw"].iloc[0] == 2.5
    # Labels cannot be shifted.
    assert result["SepsisLabel"].tolist() == [0, 0, 0, 1, 1]
    # Reversing input row order must not change chronological output.
    reversed_input = make_patient().iloc[::-1]
    reordered = preprocess_patient(reversed_input)
    pd.testing.assert_frame_equal(result, reordered)
    # No other patient's values can flow into a separately processed patient.
    second = make_patient()
    second["HR"] = np.nan
    assert preprocess_patient(second)["HR"].isna().all()
    try:
        duplicate = make_patient()
        duplicate.loc[1, "ICULOS"] = 1
        preprocess_patient(duplicate)
        raise AssertionError("Duplicate times should be rejected")
    except ValueError:
        pass
    print("PASS: causal, age-limited fill, masks, counts, and labels")


if __name__ == "__main__":
    main()
