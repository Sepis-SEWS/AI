"""AI v2-10: sequential LSTM weight + sampling experiments, locked Validation.
Run from repo root. Existing checkpoints and histories are never overwritten.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src/training/train_lstm.py"
MANIFEST = ROOT / "data/splits/patient_splits.json"
STATS = ROOT / "models/lstm_preprocessing_stats.json"
CACHE = ROOT / "data/cache/lstm_v2"
# Start with the best single-model weight and vary sampling independently.
CONFIGS = {
    "pw3_pos2": (3, None, 2),
    "pw3_neg10": (3, 10, 1),
    "pw3_pos3": (3, None, 3),
    "pw3_neg20": (3, 20, 1),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    args = parser.parse_args()
    if len(set(args.configs)) != len(args.configs):
        parser.error("Duplicate configuration")
    for p in (TRAIN, MANIFEST, STATS, ROOT / "data/splits/patient_index.csv"):
        if not p.is_file():
            parser.error(f"Missing {p}")
    split = json.loads(MANIFEST.read_text(encoding="utf-8"))["split"]
    if tuple(len(split[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        parser.error("Unexpected split counts")
    if len(set(split["train"] + split["validation"] + split["test"])) != 40336:
        parser.error("Patient split overlap")
    commands = []
    for name in args.configs:
        pw, neg, repeat = CONFIGS[name]
        tag = "v2_10_" + name
        stem = f"lstm_full_pw{pw}_{tag}"
        if neg is not None:
            stem += f"_neg{neg}"
        if repeat > 1:
            stem += f"_pos{repeat}"
        model = ROOT / "models" / (stem + "_best.pt")
        history = ROOT / "reports" / (stem + "_history.json")
        if model.exists() or history.exists():
            parser.error(f"Existing artifacts protected: {name}. Select only unstarted configs.")
        cmd = [sys.executable, str(TRAIN), "--manifest", str(MANIFEST),
               "--stats", str(STATS), "--cache-dir", str(CACHE),
               "--run-tag", tag, "--pos-weight", str(pw),
               "--epochs", "12", "--patience", "3", "--batch-size", "256",
               "--hidden-size", "64", "--seed", "42", "--device", args.device]
        if neg is not None:
            cmd += ["--negative-ratio", str(neg)]
        if repeat > 1:
            cmd += ["--positive-repeat", str(repeat)]
        commands.append((name, cmd, model, history))
    for name, cmd, _, _ in commands:
        print(name, subprocess.list2cmdline(cmd), flush=True)
    if args.dry_run:
        print("DRY RUN OK: no training; Test HELD OUT.", flush=True)
        return
    results = []
    for name, cmd, model, history in commands:
        print(f"Starting {name}", flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        if not model.is_file() or not history.is_file():
            raise RuntimeError(f"Missing output for {name}")
        report = json.loads(history.read_text(encoding="utf-8"))
        if (report["train_patients"], report["validation_patients"],
                report["test_patients_evaluated"]) != (25814, 6454, 0):
            raise ValueError("Unexpected evaluation scope")
        best = max(report["history"], key=lambda x: x["validation"]["AP"])
        results.append((name, best["epoch"], best["validation"]["AP"],
                        best["validation"]["AUROC"]))
    print("Config                         Epoch       AP      AUROC")
    for name, epoch, ap, auc in sorted(results, key=lambda r: -r[2]):
        print(f"{name:30s} {epoch:5d} {ap:9.6f} {auc:9.6f}")
    print("Test HELD OUT. Individual histories and checkpoints preserved.")

if __name__ == "__main__":
    main()
