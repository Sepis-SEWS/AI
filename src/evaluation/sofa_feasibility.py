"""Audit availability and safely computable SOFA/qSOFA *components* on PhysioNet 2019.

Usage (from repository root):
    python src/evaluation/sofa_feasibility.py --sample-patients 200
    python src/evaluation/sofa_feasibility.py

This is not a complete SOFA/qSOFA score, a sepsis diagnosis, or new labels.
No missing components are filled as zero. No patient outcomes are used.
Reference: Singer et al., JAMA 2016 (Sepsis-3), doi:10.1001/jama.2016.0287.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("data/data-sepsis/training")
SOURCE_PARTS = ("training_setA", "training_setB")
CHECKED = ("Resp", "SBP", "MAP", "Platelets", "Bilirubin_total", "Creatinine")
UNAVAILABLE = {
    "SOFA_respiratory": "PaO2 and respiratory support status absent (FiO2/O2Sat alone insufficient)",
    "SOFA_CNS": "GCS absent",
    "SOFA_full_cardiovascular": "Vasopressor agents and doses absent (MAP alone insufficient)",
    "SOFA_full_renal": "Urine output absent (creatinine-only score may underestimate renal severity)",
    "qSOFA_full": "Altered mentation/GCS absent",
    "Sepsis3_relabel": "Suspected infection and complete baseline/serial SOFA unavailable",
}


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def component_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return only measured component indicators; never a fabricated total score."""
    missing = set(CHECKED) - set(df.columns)
    if missing:
        raise ValueError(f"Required PSV columns absent: {sorted(missing)}")
    o = pd.DataFrame(index=df.index)
    for name in CHECKED:
        o[name] = finite_numeric(df[name])

    # qSOFA: one point each for RR>=22 and SBP<=100; mental status unavailable.
    # A missing vital must remain missing rather than count as normal.
    o["qsofa_resp_component"] = np.where(o["Resp"].notna(), (o["Resp"] >= 22).astype(float), np.nan)
    o["qsofa_sbp_component"] = np.where(o["SBP"].notna(), (o["SBP"] <= 100).astype(float), np.nan)
    o["qsofa_two_components_observed"] = o[["Resp", "SBP"]].notna().all(axis=1)
    o["qsofa_two_component_sum"] = o[["qsofa_resp_component", "qsofa_sbp_component"]].sum(
        axis=1, min_count=2,
    )
    # SOFA component score can be assigned for Platelets (10^3/uL) and
    # Bilirubin_total/Creatinine (mg/dL), PROVIDED source units are verified.
    # These single components cannot be summed into "total SOFA."
    for field, cuts in (
        ("Platelets", (20.0, 50.0, 100.0, 150.0)),
        ("Bilirubin_total", (1.2, 2.0, 6.0, 12.0)),
        ("Creatinine", (1.2, 2.0, 3.5, 5.0)),
    ):
        v = o[field]
        if field == "Platelets":
            # >? boundaries from Sepsis-3: >=150 ->0, <150 ->1,
            # <100 ->2, <50 ->3, <20 ->4.
            out = np.select([v < 20, v < 50, v < 100, v < 150],
                            [4, 3, 2, 1], default=0).astype(float)
        else:
            # Reference uses rounded categories. Strict comparison follows
            # the conventional lower-bound 1.2/2/6/12 etc.
            out = np.select([v >= cuts[3], v >= cuts[2], v >= cuts[1],
                             v >= cuts[0]], [4, 3, 2, 1],
                            default=0).astype(float)
        o[f"sofa_{field}_component"] = np.where(v.notna() & (v >= 0), out, np.nan)
    # MAP <70 assures at least 1 cardiovascular point, but MAP>=70
    # cannot establish 0 because vasopressor status is absent.
    o["sofa_cardiovascular_map_low_flag"] = np.where(
        o["MAP"].notna(), (o["MAP"] < 70).astype(float), np.nan,
    )
    return o


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--sample-patients", type=int, default=None,
                        help="First N files per source for quick structural audit")
    parser.add_argument("--output", type=Path, default=Path("reports/sofa_feasibility.json"))
    args = parser.parse_args()
    if args.sample_patients is not None and args.sample_patients <= 0:
        parser.error("--sample-patients must be positive")

    parts = []
    for source in SOURCE_PARTS:
        paths = sorted((args.root / source).glob("*.psv"))
        if args.sample_patients is not None:
            paths = paths[:args.sample_patients]
        if not paths:
            raise FileNotFoundError(f"No PSV patients in {args.root / source}")
        sums = {k: 0 for k in CHECKED}
        component_sums = {k: 0 for k in ("qsofa_two_components_observed",
                                          "sofa_Platelets_component",
                                          "sofa_Bilirubin_total_component",
                                          "sofa_Creatinine_component",
                                          "sofa_cardiovascular_map_low_flag")}
        n_rows = 0
        invalid = 0
        for i, path in enumerate(paths, 1):
            df = pd.read_csv(path, sep="|")
            part = component_rows(df)
            n_rows += len(part)
            for name in CHECKED:
                sums[name] += int(part[name].notna().sum())
            for name in component_sums:
                component_sums[name] += int(part[name].notna().sum() if
                                            name != "qsofa_two_components_observed" else
                                            part[name].sum())
            for name in ("Platelets", "Creatinine", "Bilirubin_total"):
                invalid += int((part[name] < 0).sum())
            if i % 1000 == 0:
                print(f"  {source}: {i:,}/{len(paths):,} patients", flush=True)
        parts.append({
            "source": source, "patients": len(paths), "rows": n_rows,
            "observed_count": sums,
            "observed_rate": {k: sums[k] / n_rows for k in CHECKED},
            "component_available_count": component_sums,
            "component_available_rate": {k: component_sums[k] / n_rows for k in component_sums},
            "negative_lab_measurement_rows": invalid,
        })
    report = {
        "title": "Sepsis-3 SOFA/qSOFA component feasibility audit",
        "scope": "per-source structural audit, without label modification or model scoring",
        "sample_patients_per_set": args.sample_patients,
        "full_sofa_calculable": False,
        "full_qsofa_calculable": False,
        "limitations": UNAVAILABLE,
        "unit_assumption": ("Platelets in 10^3/uL; Bilirubin_total and Creatinine in mg/dL "
                            "per challenge documentation. Verify units before operational use."),
        "component_output_warning": ("No total SOFA/qSOFA computed. Missing vital observations "
                                     "remain missing; absent components are not normal scores."),
        "reference": "https://doi.org/10.1001/jama.2016.0287",
        "sources": parts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"sources": parts, "limitations": UNAVAILABLE},
                     ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
