"""Unit tests for partially observable SOFA/qSOFA components.

Run from repository root:
    python src/evaluation/test_sofa_feasibility.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sofa_feasibility import component_rows


def main() -> None:
    df = pd.DataFrame({
        "Resp": [22, 21, np.nan, 23],
        "SBP": [100, 110, 90, np.nan],
        "MAP": [69, 70, np.nan, 55],
        "Platelets": [150, 99, 19, np.nan],
        "Bilirubin_total": [1.1, 2.5, 12.0, np.nan],
        "Creatinine": [1.1, 3.5, 5.1, np.nan],
    })
    result = component_rows(df)
    assert result["qsofa_resp_component"].tolist()[:2] == [1, 0]
    assert result["qsofa_sbp_component"].tolist()[:2] == [1, 0]
    assert result["qsofa_two_component_sum"].iloc[0] == 2
    assert result["qsofa_two_component_sum"].iloc[1] == 0
    assert np.isnan(result["qsofa_two_component_sum"].iloc[2])
    assert np.isnan(result["qsofa_two_component_sum"].iloc[3])
    assert result["sofa_Platelets_component"].iloc[:3].tolist() == [0, 2, 4]
    assert result["sofa_Bilirubin_total_component"].iloc[:3].tolist() == [0, 2, 4]
    assert result["sofa_Creatinine_component"].iloc[:3].tolist() == [0, 3, 4]
    assert np.isnan(result["sofa_Platelets_component"].iloc[3])
    assert result["sofa_cardiovascular_map_low_flag"].iloc[0] == 1
    assert result["sofa_cardiovascular_map_low_flag"].iloc[1] == 0
    assert np.isnan(result["sofa_cardiovascular_map_low_flag"].iloc[2])
    assert not hasattr(result, "full_sofa")
    print("PASS: Sepsis-3 partial components and missing-values preservation")


if __name__ == "__main__":
    main()
