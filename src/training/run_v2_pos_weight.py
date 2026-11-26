"""SEWS v2 LSTM positive-class weight experiment runner.

Run from repository root:
    python src/training/run_v2_pos_weight.py --dry-run
    python src/training/run_v2_pos_weight.py --weights 3
    python src/training/run_v2_pos_weight.py --weights 20

Uses the ORIGINAL locked Train/Validation only (no Test).
Every experiment has a unique output filename and preflight overwrite guard.
This runner does not change the v1 trained model or its evaluation policy.
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
WEIGHTS = (3, 20)


def paths_for(weight: int) -> tuple[Path, Path, str]:
    tag = f"v2pw{weight}"
    stem = f"lstm_full_pw{weight}_{tag}"
    return ROOT / "models" / f"{stem}_best.pt", ROOT / "reports" / f"{stem}_history.json", tag


def main() -> None:
    parser = argparse.ArgumentParser(description="SEWS v2 LSTM class-weight experiment")
    parser.add_argument("--weights", type=int, nargs="+", choices=WEIGHTS, default=[3],
                        help="Positive-class weights to train sequentially (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and print commands only")
    parser.add_argument("--fast", action="store_true",
                        help="Use reusable memory-mapped Train/Validation sequence cache")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="PyTorch device (auto prefers CUDA when available)")
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="Optional CPU compute thread count for benchmarking")
    args = parser.parse_args()

    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    weights = list(args.weights)
    if len(set(weights)) != len(weights):
        parser.error("Duplicate weight specified")
    for required in (TRAIN_SCRIPT, MANIFEST, STATS, BASELINE):
        if not required.is_file():
            parser.error(f"Required file is missing: {required.relative_to(ROOT)}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = manifest["split"]
    counts = tuple(len(split[key]) for key in ("train", "validation", "test"))
    if counts != (25814, 6454, 8068):
        parser.error(f"Unexpected locked split counts: {counts}")
    if (set(split["train"]) & set(split["validation"]) or
            set(split["train"]) & set(split["test"]) or
            set(split["validation"]) & set(split["test"])):
        parser.error("Patient IDs overlap across locked splits")

    commands = []
    for weight in weights:
        checkpoint, report, tag = paths_for(weight)
        if checkpoint.exists() or report.exists():
            parser.error(
                f"Experiment output already exists for pos_weight={weight}. "
                f"Refusing to overwrite: {checkpoint.relative_to(ROOT)} / "
                f"{report.relative_to(ROOT)}"
            )
        command = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--manifest", str(MANIFEST),
            "--stats", str(STATS),
            "--pos-weight", str(weight),
            "--run-tag", tag,
            "--epochs", "12",
            "--patience", "3",
            "--batch-size", "256",
            "--seed", "42",
            "--device", args.device,
        ]
        if args.fast:
            command.extend(["--cache-dir", str(ROOT / "data" / "cache" / "lstm_v2")])
        if args.torch_threads is not None:
            command.extend(["--torch-threads", str(args.torch_threads)])
        commands.append(command)

    print("Locked patient split: Train 25,814 / Validation 6,454 / Test 8,068")
    print("v1 pos_weight=10 is the baseline; Test will not be accessed.")
    for cmd in commands:
        print("Command:", subprocess.list2cmdline(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=ROOT)
    if args.dry_run:
        print("DRY RUN OK: no training was started.")
    else:
        print("Experiment(s) finished. Compare validation AP and AUROC in reports/.")


if __name__ == "__main__":
    main()
