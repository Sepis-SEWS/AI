
from pathlib import Path
from collections import Counter
import pandas as pd

ROOT = Path("data/data-sepsis/training")
OUTPUT = Path("reports/data_validation.csv")

EXPECTED_COLUMNS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
    "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH",
    "PaCO2", "SaO2", "AST", "BUN", "Alkalinephos",
    "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate",
    "Potassium", "Bilirubin_total", "TroponinI",
    "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2",
    "HospAdmTime", "ICULOS", "SepsisLabel"
]


def validate_patient(path: Path, source: str) -> dict:
    """환자 PSV 파일의 구조와 시간, 라벨을 검사한다."""
    result = {
        "source": source,
        "patient_id": path.stem,
        "file": str(path),
        "rows": 0,
        "positive": 0,
        "positive_hours": 0,
        "status": "OK",
        "errors": ""
    }

    errors = []

    try:
        df = pd.read_csv(path, sep="|")
        result["rows"] = len(df)

        if df.empty:
            errors.append("EMPTY_FILE")

        if df.columns.tolist() != EXPECTED_COLUMNS:
            errors.append("INVALID_COLUMNS")

        if "ICULOS" in df.columns:
            hours = pd.to_numeric(
                df["ICULOS"], errors="coerce"
            )

            if hours.isna().any():
                errors.append("INVALID_ICULOS")

            if hours.duplicated().any():
                errors.append("DUPLICATE_ICULOS")

            if not hours.is_monotonic_increasing:
                errors.append("UNSORTED_ICULOS")
        else:
            errors.append("MISSING_ICULOS")

        if "SepsisLabel" in df.columns:
            labels = pd.to_numeric(
                df["SepsisLabel"], errors="coerce"
            )

            if not labels.isin([0, 1]).all():
                errors.append("INVALID_LABEL")
            else:
                result["positive"] = int(labels.max())
                result["positive_hours"] = int(
                    labels.sum()
                )
        else:
            errors.append("MISSING_LABEL")

    except Exception as exc:
        errors.append(f"READ_ERROR: {exc}")

    if errors:
        result["status"] = "ERROR"
        result["errors"] = "; ".join(errors)

    return result


def main() -> None:
    """Set A/B 전체 환자를 검사하고 보고서를 저장한다."""
    records = []
    counts = Counter()

    for source in ["training_setA", "training_setB"]:
        files = sorted((ROOT / source).glob("*.psv"))
        print(f"\n{source}: {len(files):,} files")

        for index, path in enumerate(files, start=1):
            record = validate_patient(path, source)
            records.append(record)
            counts[record["status"]] += 1

            if index % 2000 == 0:
                print(f"  checked {index:,}/{len(files):,}")

    report = pd.DataFrame(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\n=== Validation summary ===")
    print("Total patients:", len(report))
    print("Normal files:", counts["OK"])
    print("Error files:", counts["ERROR"])
    print("Positive patients:", report["positive"].sum())
    print("Positive hours:", report["positive_hours"].sum())
    print("\nBy source:")
    print(
        report.groupby("source")[
            ["rows", "positive", "positive_hours"]
        ].sum()
    )
    print("\nReport:", OUTPUT)


if __name__ == "__main__":
    main()
