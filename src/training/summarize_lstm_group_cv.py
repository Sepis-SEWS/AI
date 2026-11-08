"""Summarize all five completed SEWS LSTM Group-CV folds (TRAIN only).

Run from repository root after fold1..fold5 finish:
    python src/training/summarize_lstm_group_cv.py

Reads saved training histories and fold manifests only. It does NOT load PSV,
score patients, modify checkpoints or touch locked Validation/Test.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))
from lstm_sequences import read_manifest  # noqa: E402

ORIGINAL = Path("data/splits/patient_splits.json")
FOLDS = Path("data/splits/lstm_cv")
REPORTS = Path("reports")
OUT = REPORTS / "lstm_group5fold_summary.json"
EPOCH_LIMIT = 12
PATIENCE = 3


def main() -> None:
    groups = read_manifest(ORIGINAL)
    if [len(groups[k]) for k in ("train", "validation", "test")] != [25814, 6454, 8068]:
        raise ValueError("Original locked patient split counts changed")
    original_train = set(groups["train"])
    forbidden = set(groups["validation"]) | set(groups["test"])
    if original_train & forbidden:
        raise ValueError("Original splits overlap")

    rows = []
    heldout_sets = []
    for fold in range(1, 6):
        tag = f"fold{fold}"
        manifest_path = FOLDS / f"{tag}.json"
        history_path = REPORTS / f"lstm_full_pw10_{tag}_history.json"
        checkpoint_path = Path("models") / f"lstm_full_pw10_{tag}_best.pt"
        if not manifest_path.exists() or not history_path.exists():
            raise FileNotFoundError(
                f"Fold {fold} not finished or not available locally: "
                f"{manifest_path}, {history_path}"
            )
        split = read_manifest(manifest_path)
        fit, hold = set(split["train"]), set(split["validation"])
        if (len(fit) != len(split["train"]) or len(hold) != len(split["validation"])
                or fit & hold or fit | hold != original_train
                or (fit | hold) & forbidden):
            raise ValueError(f"Invalid group partition in {manifest_path}")
        if fold < 5 and (len(fit), len(hold)) != (20651, 5163):
            raise ValueError(f"Unexpected patient counts in {manifest_path}")
        if fold == 5 and (len(fit), len(hold)) != (20652, 5162):
            raise ValueError(f"Unexpected patient counts in {manifest_path}")
        heldout_sets.append(hold)
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if (history.get("run_tag") != tag or
                float(history.get("pos_weight", -1)) != 10.0 or
                history.get("test_patients_evaluated") != 0 or
                history.get("train_patients") != len(fit) or
                history.get("validation_patients") != len(hold)):
            raise ValueError(f"Fold metadata inconsistent: {history_path}")
        if Path(str(history.get("manifest", "")).replace("\\", "/")).parts != manifest_path.parts:
            raise ValueError(f"Fold manifest path mismatch: {history_path}")
        epochs = history.get("history", [])
        if not 1 <= len(epochs) <= EPOCH_LIMIT:
            raise ValueError(f"Unexpected epoch history: {history_path}")
        if [entry["epoch"] for entry in epochs] != list(range(1, len(epochs) + 1)):
            raise ValueError(f"Epoch sequence inconsistent: {history_path}")
        best = max(epochs, key=lambda row: float(row["validation"]["AP"]))
        best_index = best["epoch"] - 1
        # Distinguish fully completed CV runs from an in-progress history JSON.
        if len(epochs) < EPOCH_LIMIT and len(epochs) - best["epoch"] < PATIENCE:
            raise RuntimeError(
                f"Fold {fold} history appears IN PROGRESS (epoch {len(epochs)}, "
                f"best epoch {best['epoch']}). Wait for its final 'Saved model' output."
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint for fold {fold} missing: {checkpoint_path}. "
                "Do not substitute another fold's model."
            )
        if not math.isclose(float(history["best_validation_AP"]),
                            float(best["validation"]["AP"]), rel_tol=1e-10):
            raise ValueError(f"Best AP in history inconsistent: {history_path}")
        rows.append({
            "fold": fold, "train_patients": len(fit),
            "heldout_train_patients": len(hold),
            "best_epoch_by_AP": int(best["epoch"]),
            "epochs_completed": len(epochs),
            "AUROC_at_best_AP_epoch": float(best["validation"]["AUROC"]),
            "AP_at_best_AP_epoch": float(best["validation"]["AP"]),
            "validation_scoreable_rows": int(best["validation"]["samples"]),
            "validation_positive_rate": float(best["validation"]["positive_rate"]),
            "history_file": str(history_path).replace("\\", "/"),
            "checkpoint_file": str(checkpoint_path).replace("\\", "/"),
        })
    if len(set.union(*heldout_sets)) != len(original_train):
        raise ValueError("Fold held-out union does not match original Train")
    if sum(map(len, heldout_sets)) != len(original_train):
        raise ValueError("A patient was held out in multiple folds")

    aurocs = np.array([r["AUROC_at_best_AP_epoch"] for r in rows])
    aps = np.array([r["AP_at_best_AP_epoch"] for r in rows])
    summary = {
        "status": "LSTM_TRAIN_ONLY_5FOLD_COMPLETED",
        "evaluation_scope": "Original locked Train patients only; each patient in one held-out fold",
        "model": "24x64 RiskLSTM; pos_weight=10; best AP checkpoint per fold",
        "folds": rows,
        "average_AUROC": float(aurocs.mean()),
        "sample_std_AUROC": float(aurocs.std(ddof=1)),
        "average_AP": float(aps.mean()),
        "sample_std_AP": float(aps.std(ddof=1)),
        "selection_note": (
            "Best epoch selected on each fold's held-out patients; "
            "this is cross-validation model selection, not an independently "
            "estimated clinical performance. Original Validation and final Test "
            "were NOT rescored or used for CV."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise FileExistsError(f"Existing 5-fold summary will not be overwritten: {OUT}")
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for row in rows:
        print(f"Fold {row['fold']}: best epoch={row['best_epoch_by_AP_epoch']}, "
              f"AUROC={row['AUROC_at_best_AP_epoch']:.6f}, "
              f"AP={row['AP_at_best_AP_epoch']:.6f}")
    print(f"5-Fold mean±SD: AUROC={summary['average_AUROC']:.6f} ± "
          f"{summary['sample_std_AUROC']:.6f}; "
          f"AP={summary['average_AP']:.6f} ± {summary['sample_std_AP']:.6f}")
    print(f"Saved: {OUT}. Test NOT accessed.")


if __name__ == "__main__":
    main()
