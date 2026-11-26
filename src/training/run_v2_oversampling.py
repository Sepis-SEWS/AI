"""SEWS AI v2-4: Train-only positive-sequence oversampling (2 copies).

This isolates oversampling against the original pos_weight=10 LSTM baseline.
Validation remains fully natural; Test is locked and never read.
GPU and preprocessed memory-mapped cache are reused; prior outputs are protected.

Usage from repository root:
  python src/training/run_v2_oversampling.py --dry-run
  python src/training/run_v2_oversampling.py --smoke-test
  python src/training/run_v2_oversampling.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "training" / "train_lstm.py"
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
STATS = ROOT / "models" / "lstm_preprocessing_stats.json"
BASELINE = ROOT / "models" / "lstm_full_pw10_best.pt"
REPEAT = 2


def main():
    parser = argparse.ArgumentParser(description="AI v2-4 train-only positive oversampling")
    parser.add_argument("--dry-run", action="store_true", help="Print command and validate inputs; no training")
    parser.add_argument("--smoke-test", action="store_true", help="300 Train/100 Val, one Epoch")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.dry_run and args.smoke_test:
        parser.error("Use --dry-run or --smoke-test, not both")
    for path in (TRAIN, MANIFEST, STATS, BASELINE):
        if not path.is_file():
            parser.error(f"Required file missing: {path}")
    groups = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    sizes = tuple(len(groups[k]) for k in ("train", "validation", "test"))
    if sizes != (25814, 6454, 8068):
        parser.error(f"Unexpected locked split sizes: {sizes}")
    if any(set(groups[a]) & set(groups[b])
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        parser.error("Patient leakage in locked manifest")
    smoke = args.smoke_test
    tag = "v2over2_smoke" if smoke else "v2over2"
    scope = "train300_val100" if smoke else "full"
    stem = f"lstm_{scope}_pw10_{tag}_pos{REPEAT}"
    checkpoint = ROOT / "models" / f"{stem}_best.pt"
    history = ROOT / "reports" / f"{stem}_history.json"
    command = [
        sys.executable, str(TRAIN),
        "--manifest", str(MANIFEST), "--stats", str(STATS),
        "--pos-weight", "10", "--positive-repeat", str(REPEAT),
        "--run-tag", tag, "--epochs", "1" if smoke else "12",
        "--patience", "3", "--batch-size", "256", "--seed", "42",
        "--device", args.device,
        "--cache-dir", str(ROOT / "data" / "cache" / "lstm_v2"),
    ]
    if smoke:
        command += ["--max-train", "300", "--max-validation", "100"]
    print("Train positives repeated twice; all negative Train windows retained.", flush=True)
    print("Validation remains natural; Test HELD OUT; previous results protected.", flush=True)
    print("Command:", subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        print("DRY RUN OK (no training).")
        if checkpoint.exists() or history.exists():
            print("Note: existing experiment outputs would prevent a real rerun.")
        return
    if checkpoint.exists() or history.exists():
        parser.error(f"Existing experiment output; refusing overwrite: {checkpoint} / {history}")
    subprocess.run(command, cwd=ROOT, check=True)
    print("Completed AI v2-4. Compare highest Validation AP with 0.1354 and 0.1422.", flush=True)


if __name__ == "__main__":
    main()
