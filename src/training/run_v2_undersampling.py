"""SEWS v2-3: isolated train-only negative-window undersampling experiment.

Run after GitHub Desktop pull:
    python src/training/run_v2_undersampling.py --dry-run
    python src/training/run_v2_undersampling.py --smoke-test
    python src/training/run_v2_undersampling.py

Train: retain all positive windows, sample negatives 10:1 with at least one
negative window per eligible patient. Validation: original unmodified distribution.
Model: original pos_weight=10 baseline. Test is never accessed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "src" / "training" / "train_lstm.py"
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
STATS = ROOT / "models" / "lstm_preprocessing_stats.json"
BASELINE = ROOT / "models" / "lstm_full_pw10_best.pt"
RATIO = 10


def main():
    parser = argparse.ArgumentParser(description="SEWS v2 train-only negative undersampling")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true",
                        help="300 Train / 100 Validation / 1 Epoch isolated check")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.dry_run and args.smoke_test:
        parser.error("Choose one of --dry-run and --smoke-test")
    for required in (TRAIN_SCRIPT, MANIFEST, STATS, BASELINE):
        if not required.is_file():
            parser.error(f"Missing required file: {required}")
    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    counts = tuple(len(groups[k]) for k in ("train", "validation", "test"))
    if counts != (25814, 6454, 8068):
        parser.error(f"Locked split changed: {counts}")
    if any(set(groups[a]) & set(groups[b])
           for a, b in (("train", "validation"), ("train", "test"),
                        ("validation", "test"))):
        parser.error("Patient ID leakage across locked splits")
    tag = "v2under10_smoke" if args.smoke_test else "v2under10"
    scope = "train300_val100" if args.smoke_test else "full"
    stem = f"lstm_{scope}_pw10_{tag}_neg{RATIO}"
    out, report = ROOT / "models" / f"{stem}_best.pt", ROOT / "reports" / f"{stem}_history.json"
    if out.exists() or report.exists():
        parser.error(f"Existing output will not be overwritten: {out} or {report}")
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--manifest", str(MANIFEST), "--stats", str(STATS),
        "--pos-weight", "10", "--negative-ratio", str(RATIO),
        "--run-tag", tag,
        "--epochs", "1" if args.smoke_test else "12",
        "--patience", "3", "--batch-size", "256", "--seed", "42",
        "--device", args.device,
        "--cache-dir", str(ROOT / "data" / "cache" / "lstm_v2")
    ]
    if args.smoke_test:
        cmd.extend(["--max-train", "300", "--max-validation", "100"])
    print("Train-only undersampling 10 negatives per positive; full Validation unchanged", flush=True)
    print("Test held out; original baseline and previous v2 experiments protected", flush=True)
    print("Command:", subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        print("DRY RUN OK: no training started")
        return
    subprocess.run(cmd, cwd=ROOT, check=True)
    print("Completed; compare Validation AP against pos_weight 10 and 3.", flush=True)


if __name__ == "__main__":
    main()
