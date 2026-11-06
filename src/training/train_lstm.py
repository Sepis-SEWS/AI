"""SEWS AI-5: LSTM training on causal 24x64 patient sequences.

Run from repository root:
  python -m pip install torch
  python src/training/train_lstm.py --max-train 300 --max-validation 100 --epochs 2
  python src/training/train_lstm.py --epochs 12

Requires models/lstm_preprocessing_stats.json fitted on all locked Train patients.
Validation is read only for evaluation and early stopping; Test is never accessed.
For full runs, patient files are streamed per epoch; no all-patient tensors in RAM.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "preprocessing"))

from lstm_sequences import FEATURES, make_sequences, patient_path, read_manifest  # noqa: E402
from train_xgboost_manifest import subset  # noqa: E402

DEFAULT_MANIFEST = Path("data/splits/patient_splits.json")
DEFAULT_INDEX = Path("data/splits/patient_index.csv")
DEFAULT_STATS = Path("models/lstm_preprocessing_stats.json")
DEFAULT_ROOT = Path("data/data-sepsis/training")


class PatientSequences(IterableDataset):
    def __init__(self, ids: list[str], root: Path, stats: dict, batch_size: int,
                 shuffle: bool, seed: int):
        self.ids, self.root, self.stats = ids, root, stats
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.epoch = 0

    def __iter__(self):
        order = list(self.ids)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        # Buffers are limited to one patient plus one minibatch.
        buf_x, buf_y = [], []
        for patient_id in order:
            df = pd.read_csv(patient_path(self.root, patient_id), sep="|")
            x, y = make_sequences(df, self.stats)
            if not len(y):
                continue
            if self.shuffle:
                perm = rng.permutation(len(y))
                x, y = x[perm], y[perm]
            for row, label in zip(x, y):
                buf_x.append(row)
                buf_y.append(label)
                if len(buf_y) >= self.batch_size:
                    yield (torch.from_numpy(np.stack(buf_x)),
                           torch.from_numpy(np.asarray(buf_y, dtype=np.float32)))
                    buf_x, buf_y = [], []
        if buf_y:
            yield (torch.from_numpy(np.stack(buf_x)),
                   torch.from_numpy(np.asarray(buf_y, dtype=np.float32)))


class RiskLSTM(nn.Module):
    def __init__(self, hidden: int = 64, layers: int = 1, dropout: float = 0.2):
        super().__init__()
        # LSTM's dropout works only between multiple recurrent layers.
        self.lstm = nn.LSTM(input_size=len(FEATURES) * 4, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def run_epoch(model, batches, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    predictions, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            n += len(y)
            if not train:
                predictions.append(torch.sigmoid(logits).cpu().numpy())
                labels.append(y.cpu().numpy())
    if n == 0:
        raise ValueError("No 24-row windows in this cohort")
    result = {"loss": total_loss / n, "samples": n}
    if not train:
        yy = np.concatenate(labels)
        pp = np.concatenate(predictions)
        if len(np.unique(yy)) < 2:
            raise ValueError("Validation lacks one of the two classes")
        result.update({"AUROC": float(roc_auc_score(yy, pp)),
                       "AP": float(average_precision_score(yy, pp)),
                       "positive_rate": float(np.mean(yy))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-validation", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Weight for positive samples in BCE loss (1.0 = baseline)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-tag", type=str, default="",
                        help="Unique identifier appended to model/history filenames (required for CV)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not np.isfinite(args.pos_weight) or args.pos_weight <= 0:
        parser.error("--pos-weight must be finite and positive")

    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.run_tag):
        parser.error("--run-tag may contain only letters, numbers, _ or -")
    groups = read_manifest(args.manifest)
    is_cv = args.manifest.parent.name == "lstm_cv"
    if is_cv and not re.fullmatch(r"fold[1-5]", args.run_tag):
        parser.error("LSTM CV folds require --run-tag fold1..fold5 to prevent overwrites")
    if is_cv and args.stats.name != f"lstm_cv_{args.run_tag}_stats.json":
        parser.error("CV fold-specific --stats path does not match --run-tag")
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    if (stats.get("train_patients_used") != len(groups["train"]) or
            stats.get("train_patients_in_manifest") != len(groups["train"])):
        raise ValueError("LSTM stats must be fitted on ALL locked Train patients; do not use sampled stats")
    if stats["features"] != FEATURES or stats["window_rows"] != 24:
        raise ValueError("LSTM features/window schema mismatch")
    index = pd.read_csv(args.index)
    if index["patient_id"].duplicated().any():
        raise ValueError("Duplicate patient IDs")
    train_ids = subset(groups["train"], index, args.max_train, args.seed)
    val_ids = subset(groups["validation"], index, args.max_validation, args.seed + 1)
    if (set(train_ids) & set(val_ids) or
            (set(train_ids) | set(val_ids)) & set(groups["test"])):
        raise ValueError("Patient leakage detected")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available")
    scope = ("full" if len(train_ids) == len(groups["train"]) and
             len(val_ids) == len(groups["validation"]) else
             f"train{len(train_ids)}_val{len(val_ids)}")
    # Separate artifact names prevent different loss experiments overwriting each other.
    variant = f"_pw{args.pos_weight:g}" if args.pos_weight != 1.0 else ""
    if args.run_tag:
        variant += f"_{args.run_tag}"
    if is_cv and (Path("models") / f"lstm_{scope}{variant}_best.pt").exists():
        raise FileExistsError("CV checkpoint already exists; refusing to overwrite")
    out = Path("models") / f"lstm_{scope}{variant}_best.pt"
    report_path = Path("reports") / f"lstm_{scope}{variant}_history.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model = RiskLSTM(hidden=args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Use unweighted BCE first; alternative weighting can be compared later.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(args.pos_weight, device=device, dtype=torch.float32))
    train_data = PatientSequences(train_ids, args.data_root, stats, args.batch_size,
                                  True, args.seed)
    val_data = PatientSequences(val_ids, args.data_root, stats, args.batch_size,
                                False, args.seed)
    history = []
    best, bad_epochs = -1.0, 0
    print(f"Training LSTM {scope} on {device}, pos_weight={args.pos_weight:g}, Test HELD OUT; "
          f"Train {len(train_ids)}, Val {len(val_ids)} patients", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_data.epoch = epoch
        training = run_epoch(model, train_data, device, criterion, optimizer)
        validation = run_epoch(model, val_data, device, criterion)
        row = {"epoch": epoch, "training": training, "validation": validation}
        history.append(row)
        print(f"Epoch {epoch}: loss={training['loss']:.4f} "
              f"Val AUROC={validation['AUROC']:.4f} AP={validation['AP']:.4f}", flush=True)
        if validation["AP"] > best:
            best, bad_epochs = validation["AP"], 0
            torch.save({"model_state_dict": model.state_dict(),
                        "features": FEATURES, "input_shape": [24, len(FEATURES) * 4],
                        "stats_path": str(args.stats), "hidden_size": args.hidden_size,
                        "layers": 1, "best_epoch": epoch,
                        "validation": validation, "scope": scope,
                        "pos_weight": args.pos_weight,
                        "run_tag": args.run_tag, "manifest": str(args.manifest)}, out)
        else:
            bad_epochs += 1
        report_path.write_text(json.dumps({
            "scope": scope, "train_patients": len(train_ids),
            "validation_patients": len(val_ids), "test_patients_evaluated": 0,
            "best_validation_AP": best, "history": history,
            "pos_weight": args.pos_weight, "run_tag": args.run_tag,
            "manifest": str(args.manifest),
            "limitations": "One CV fold or single fixed Validation; not calibrated or clinical.",
        }, indent=2), encoding="utf-8")
        if bad_epochs >= args.patience:
            print(f"Early stopping after {bad_epochs} unimproved epochs", flush=True)
            break
    print(f"Saved model: {out}")
    print(f"Saved history: {report_path}")
    print("Test data never used.")


if __name__ == "__main__":
    main()
