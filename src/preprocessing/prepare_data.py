"""Create reproducible patient-wise splits and Set A/B EDA for SEWS.

Usage:
  python src/preprocessing/prepare_data.py
  python src/preprocessing/prepare_data.py --max-patients 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path("data/data-sepsis/training")
OUT = Path("data/splits")
REPORT = Path("reports")
FEATURES = [
    "HR", "Temp", "SBP", "DBP", "MAP", "Resp", "O2Sat",
    "WBC", "Lactate", "Creatinine", "Bilirubin_total",
    "Platelets", "FiO2", "BUN", "Glucose", "pH",
]
SEED = 42


def collect_files(limit: int | None) -> list[tuple[str, Path]]:
    """Enumerate and optionally reproducibly sample patients from each set."""
    rng = np.random.default_rng(SEED)
    by_set: dict[str, list[Path]] = {}
    for part in ("training_setA", "training_setB"):
        by_set[part] = sorted((ROOT / part).glob("*.psv"))
    files = [(part, path) for part, group in by_set.items() for path in group]
    if not files:
        raise FileNotFoundError(f"No PSV files found beneath {ROOT.resolve()}")
    if limit is not None and limit < len(files):
        indices = rng.choice(len(files), size=limit, replace=False)
        files = [files[i] for i in sorted(indices)]
    return files


def patient_summary(part: str, path: Path) -> dict[str, object]:
    """Read a complete patient and calculate metadata; no imputation is done."""
    df = pd.read_csv(path, sep="|")
    required = set(FEATURES) | {"SepsisLabel", "ICULOS"}
    if missing := required - set(df.columns):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: empty PSV")
    if df["SepsisLabel"].isna().any() or not df["SepsisLabel"].isin([0, 1]).all():
        raise ValueError(f"{path}: invalid SepsisLabel values")
    if df["ICULOS"].isna().any() or df["ICULOS"].duplicated().any():
        raise ValueError(f"{path}: missing or duplicate ICULOS")
    record: dict[str, object] = {
        "path": path.as_posix(),
        "source": part,
        "patient_id": f"{part}/{path.stem}",
        "positive": int(df["SepsisLabel"].max()),
        "hours": len(df),
        "positive_hours": int(df["SepsisLabel"].sum()),
    }
    for name in FEATURES:
        v = pd.to_numeric(df[name], errors="coerce")
        record[f"{name}_missing"] = float(v.isna().mean())
        record[f"{name}_observed"] = int(v.notna().sum())
        record[f"{name}_mean"] = float(v.mean()) if v.notna().any() else np.nan
    return record


def partition(meta: pd.DataFrame) -> dict[str, list[str]]:
    """Split disjoint patients, stratifying jointly by source and positivity."""
    if len(meta) < 20:
        raise ValueError("Need at least 20 patients to make meaningful splits.")
    labels = meta["source"].astype(str) + "_" + meta["positive"].astype(str)
    if labels.value_counts().min() < 3:
        raise ValueError(
            "Some Set A/B × positive strata have fewer than 3 patients; "
            "increase --max-patients or use full data."
        )
    ids = meta["patient_id"].to_numpy()
    dev, test = train_test_split(ids, test_size=0.20, random_state=SEED,
                                 stratify=labels.to_numpy())
    table = meta.set_index("patient_id")
    dev_labels = (table.loc[dev, "source"].astype(str) + "_" +
                  table.loc[dev, "positive"].astype(str)).to_numpy()
    train, validation = train_test_split(
        dev, test_size=0.20, random_state=SEED, stratify=dev_labels
    )
    groups = {"train": list(train), "validation": list(validation),
              "test": list(test)}
    assert sum(len(v) for v in groups.values()) == len(meta)
    assert not (set(train) & set(validation) or set(train) & set(test)
                or set(validation) & set(test))
    return groups


def main() -> None:
    """Write a split manifest and descriptive EDA; never fit on test data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-patients", type=int, default=None)
    args = parser.parse_args()
    records = []
    files = collect_files(args.max_patients)
    for count, (source, path) in enumerate(files, 1):
        records.append(patient_summary(source, path))
        if count % 1000 == 0:
            print(f"Read {count:,}/{len(files):,} patients")
    meta = pd.DataFrame.from_records(records)
    if meta["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient identifiers")
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUT / "patient_index.csv", index=False)
    groups = partition(meta)
    manifest = {"seed": SEED, "source_root": ROOT.as_posix(),
                "sampled_patients": args.max_patients,
                "split": groups}
    # A sampled smoke-test manifest must not overwrite the full-data manifest.
    suffix = f"_sample{args.max_patients}" if args.max_patients else ""
    target = OUT / f"patient_splits{suffix}.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    cols = (["positive", "hours", "positive_hours"] +
            [f"{v}_{stat}" for v in FEATURES
             for stat in ("missing", "observed", "mean")])
    eda = meta.groupby("source")[cols].mean()
    eda["patients"] = meta.groupby("source").size()
    eda.to_csv(REPORT / f"set_ab_eda{suffix}.csv")
    print(f"Saved manifest: {target} (seed={SEED})")
    for group, patient_ids in groups.items():
        block = meta.set_index("patient_id").loc[patient_ids]
        print(f"{group}: {len(block):,} patients, "
              f"positive patients={block['positive'].sum():,}")
        print("  sources:", block["source"].value_counts().to_dict())
    print("Set A/B descriptive comparison:", REPORT / f"set_ab_eda{suffix}.csv")
    print("Interpret source differences before merging. "
          "This is descriptive EDA, not a domain-shift significance test.")


if __name__ == "__main__":
    main()
