"""Verify cached batches exactly match the original SEWS LSTM loader.

Usage (from repo root):
    python src/training/check_lstm_cache.py

Only a small selection from locked Train and Validation is processed.
The locked Test split is never opened. Existing model files are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lstm_sequences import read_manifest
from train_lstm import CachedPatientSequences, PatientSequences


def check(split, ids, root, stats, cache_dir, shuffle):
    batch_size = 16
    old = PatientSequences(ids, root, stats, batch_size, shuffle, 42)
    new = CachedPatientSequences(ids, root, stats, batch_size, shuffle, 42,
                                 cache_dir, split)
    for epoch in (1, 2):
        old.epoch = new.epoch = epoch
        a, b = iter(old), iter(new)
        count = 0
        while True:
            try:
                original = next(a)
            except StopIteration:
                try:
                    next(b)
                except StopIteration:
                    break
                raise AssertionError("Cache yielded additional batches")
            try:
                cached = next(b)
            except StopIteration:
                raise AssertionError("Cache yielded fewer batches")
            for old_tensor, new_tensor in zip(original, cached):
                if not np.array_equal(old_tensor.numpy(), new_tensor.numpy()):
                    raise AssertionError(f"{split} epoch {epoch}: cached batch mismatch")
            count += len(original[1])
        print(f"PASS {split} epoch {epoch}: {count:,} identical sequences", flush=True)


def main():
    root = Path("data/data-sepsis/training")
    manifest = read_manifest(Path("data/splits/patient_splits.json"))
    stats = json.loads(Path("models/lstm_preprocessing_stats.json").read_text(encoding="utf-8"))
    cache_dir = Path("data/cache/lstm_v2_smoke")
    for split, shuffle in (("train", True), ("validation", False)):
        ids = manifest[split][:12]
        check(split, ids, root, stats, cache_dir, shuffle)
    print("PASS: old and cached loaders match. Test not accessed.", flush=True)


if __name__ == "__main__":
    main()
