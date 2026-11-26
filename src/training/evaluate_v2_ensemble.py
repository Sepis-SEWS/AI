"""AI v2-5: compare frozen v2 best LSTM with original XGBoost on Validation.

Does not retrain either model, does not read Test patients, and does not overwrite
the v1 ensemble reports. Uses previous aligned XGBoost Validation score archive,
then verifies its patient IDs and labels against the locked manifest and v2 cache.
Run from repository root after GitHub Desktop pull:
    python src/training/evaluate_v2_ensemble.py --dry-run
    python src/training/evaluate_v2_ensemble.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))
from lstm_sequences import FEATURES, read_manifest  # noqa: E402
from lstm_cache import load_or_build, WINDOW_SHAPE  # noqa: E402
from train_lstm import RiskLSTM  # noqa: E402

ROOT = HERE.parents[1]
MANIFEST = ROOT / "data" / "splits" / "patient_splits.json"
STATS = ROOT / "models" / "lstm_preprocessing_stats.json"
CHECKPOINT = ROOT / "models" / "lstm_full_pw3_v2pw3_retry1_best.pt"
PREVIOUS = ROOT / "reports" / "ensemble_validation_scores.npz"
CACHE = ROOT / "data" / "cache" / "lstm_v2"
RESULTS = ROOT / "reports" / "v2_ensemble_validation_comparison.csv"
SUMMARY = ROOT / "reports" / "v2_ensemble_validation_comparison.json"
ARCHIVE = ROOT / "reports" / "v2_ensemble_validation_scores.npz"
WEIGHTS = (0.0, 0.5, 0.75, 0.9, 1.0)


def _metrics(y, scores):
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be finite and between 0 and 1")
    return {"AUROC": float(roc_auc_score(y, scores)),
            "AP": float(average_precision_score(y, scores))}


def main():
    parser = argparse.ArgumentParser(description="V2 locked Validation XGBoost/LSTM ensemble")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for path in (MANIFEST, STATS, CHECKPOINT, PREVIOUS):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    groups = read_manifest(MANIFEST)
    if tuple(len(groups[k]) for k in ("train", "validation", "test")) != (25814, 6454, 8068):
        parser.error("Original locked patient split changed")
    for path in (RESULTS, SUMMARY, ARCHIVE):
        if path.exists() and not args.dry_run:
            parser.error(f"Existing v2 result protected from overwrite: {path}")
    if args.dry_run:
        print("DRY RUN OK: read only locked Validation archive and v2 LSTM checkpoint.")
        print("Output:", RESULTS, SUMMARY, ARCHIVE)
        print("Test is never accessed; original v1 ensemble outputs stay unchanged.")
        return

    stats = json.loads(STATS.read_text(encoding="utf-8"))
    if (stats.get("train_patients_used") != 25814 or
            stats.get("train_patients_in_manifest") != 25814 or
            stats.get("features") != FEATURES or stats.get("window_rows") != 24):
        raise ValueError("Expected original full-Train preprocessing stats")
    cache_path, meta = load_or_build(CACHE, "validation", groups["validation"],
                                     ROOT / "data" / "data-sepsis" / "training", stats)
    n = meta["samples"]
    labels_cache = np.memmap(cache_path / "y.f32", dtype=np.float32, mode="r", shape=(n,))
    patient_ids = np.repeat(np.asarray(groups["validation"], dtype=str),
                            np.diff(meta["offsets"]))
    if len(patient_ids) != n:
        raise ValueError("Cache offsets do not match manifest")

    with np.load(PREVIOUS, allow_pickle=False) as existing:
        required = {"patient_ids", "labels", "xgboost_scores", "lstm_scores", "icul_os"}
        if not required.issubset(existing.files):
            raise ValueError(f"Missing original aligned score fields: {required - set(existing.files)}")
        baseline_ids = existing["patient_ids"].astype(str)
        y = existing["labels"].astype(np.int8)
        xgb = existing["xgboost_scores"].astype(np.float64)
        v1_lstm = existing["lstm_scores"].astype(np.float64)
        times = existing["icul_os"].copy()
    if not (len(y) == len(xgb) == len(v1_lstm) == len(patient_ids) == len(times) == n):
        raise ValueError("Archived scores and cache have different row counts")
    if not np.array_equal(patient_ids, baseline_ids):
        raise ValueError("Archived XGBoost patient ordering differs from v2 cache")
    if not np.array_equal(y, labels_cache.astype(np.int8)):
        raise ValueError("Archived XGBoost target labels differ from v2 cache")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Invalid Validation labels")
    for score in (xgb, v1_lstm):
        _metrics(y, score)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (checkpoint.get("scope") != "full" or checkpoint.get("input_shape") != [24, 64] or
            checkpoint.get("features") != FEATURES or
            float(checkpoint.get("pos_weight", -1)) != 3.0 or
            checkpoint.get("run_tag") != "v2pw3_retry1" or
            int(checkpoint.get("best_epoch", -1)) != 10):
        raise ValueError("Not the completed AI v2-1 best LSTM checkpoint")
    if checkpoint.get("stats_path") not in (str(STATS), "models/lstm_preprocessing_stats.json"):
        # The original training recorded a relative path; permit its Windows equivalent.
        if Path(str(checkpoint.get("stats_path"))).name != STATS.name:
            raise ValueError("Checkpoint statistics reference is unexpected")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot access the GPU")
    model = RiskLSTM(hidden=int(checkpoint["hidden_size"]),
                     layers=int(checkpoint["layers"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    x = np.memmap(cache_path / "x.f32", dtype=np.float32,
                  mode="r", shape=(n, *WINDOW_SHAPE))
    scores_parts = []
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            batch = torch.from_numpy(np.array(x[start:start + args.batch_size], copy=True)).to(device)
            scores_parts.append(torch.sigmoid(model(batch)).cpu().numpy())
            if start and start % (args.batch_size * 100) == 0:
                print(f"Scored {start:,}/{n:,} Validation sequences", flush=True)
    v2_lstm = np.concatenate(scores_parts).astype(np.float64)
    if len(v2_lstm) != n:
        raise ValueError("Wrong number of LSTM scores")
    observed = _metrics(y, v2_lstm)
    checkpoint_ap = float(checkpoint["validation"]["AP"])
    if abs(observed["AP"] - checkpoint_ap) > 0.0005:
        raise ValueError(
            f"Validation AP {observed['AP']:.6f} differs from checkpoint "
            f"{checkpoint_ap:.6f}; inspect input alignment before using results")
    print(f"PASS: v2 checkpoint AP {checkpoint_ap:.6f} reproduced as {observed['AP']:.6f}")

    rows = []
    for name, score in (("xgboost_v1", xgb),
                        ("lstm_v1_posweight10", v1_lstm),
                        ("lstm_v2_posweight3", v2_lstm)):
        rows.append({"model": name, "kind": "individual", "lstm_weight": None,
                     **_metrics(y, score)})
    for weight in WEIGHTS:
        combined = weight * v2_lstm + (1.0 - weight) * xgb
        rows.append({"model": f"v2_raw_lstm_{weight:g}", "kind": "raw_blend",
                     "lstm_weight": weight, **_metrics(y, combined)})
    # Rank blending is a retrospective diagnostic, NOT a deployable scoring rule:
    # global ranks use the entire Validation distribution.
    xr = rankdata(xgb, method="average") / n
    lr = rankdata(v2_lstm, method="average") / n
    for weight in (0.5, 0.75, 0.9):
        rows.append({"model": f"v2_rank_lstm_{weight:g}", "kind": "rank_diagnostic",
                     "lstm_weight": weight,
                     **_metrics(y, weight * lr + (1.0 - weight) * xr)})

    table = pd.DataFrame(rows)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    # Run must not replace old v1 or previously computed v2 outputs.
    table.to_csv(RESULTS, index=False)
    np.savez_compressed(ARCHIVE, patient_ids=patient_ids, icul_os=times,
                        labels=y, xgboost_scores=xgb,
                        lstm_v1_scores=v1_lstm, lstm_v2_scores=v2_lstm)
    report = {
        "scope": "original_locked_validation_only",
        "train_patients": 25814, "validation_patients": 6454,
        "validation_rows": int(n), "test_patients_evaluated": 0,
        "v2_lstm_checkpoint": str(CHECKPOINT),
        "xgboost_source_archive": str(PREVIOUS),
        "v2_best_epoch": int(checkpoint["best_epoch"]),
        "v2_checkpoint_AP": checkpoint_ap,
        "reproduced_AP": observed["AP"],
        "results": rows,
        "limitations": (
            "Raw model scores are not calibrated probabilities. XGBoost scores are "
            "reused from the prior aligned Validation score archive. Weight selection "
            "on the same Validation is optimistically biased. Rank ensemble uses all "
            "Validation ranks and is not directly deployable. No Test used.")
    }
    SUMMARY.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(table.to_string(index=False), flush=True)
    print(f"Saved {RESULTS}, {SUMMARY}, {ARCHIVE}; Test HELD OUT.", flush=True)


if __name__ == "__main__":
    main()
