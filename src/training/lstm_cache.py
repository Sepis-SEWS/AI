"""Reusable, patient-ordered memory-mapped SEWS LSTM sequence cache.

Builds from the same make_sequences() function as the original streaming loader.
Only requested Train/Validation IDs are read; Test is never materialized.
Keep source PSV data immutable while using a cache. To refresh changed raw data,
delete the cache directory and rebuild it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lstm_sequences import FEATURES, make_sequences, patient_path

CACHE_VERSION = 1
WINDOW_SHAPE = (24, len(FEATURES) * 4)


def _identity(ids, root, stats, split):
    payload = json.dumps({
        "version": CACHE_VERSION, "split": split, "ids": ids,
        "root": str(root.resolve()), "stats": stats,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_build(cache_dir: Path, split: str, ids: list[str], root: Path, stats: dict):
    """Build one immutable cache for an exact ID list, preprocessing spec and split."""
    if split not in ("train", "validation"):
        raise ValueError("Caching Test data is prohibited for v2 experiments")
    digest = _identity(ids, root, stats, split)
    directory = cache_dir / f"{split}_{digest[:20]}"
    metadata = directory / "meta.json"
    x_path, y_path = directory / "x.f32", directory / "y.f32"
    if metadata.exists():
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        n = meta["samples"]
        expected_x = n * WINDOW_SHAPE[0] * WINDOW_SHAPE[1] * 4
        if (meta.get("digest") != digest or
                len(meta.get("offsets", [])) != len(ids) + 1 or
                not x_path.is_file() or x_path.stat().st_size != expected_x or
                not y_path.is_file() or y_path.stat().st_size != n * 4):
            raise RuntimeError(f"Corrupt LSTM cache: {directory}; remove it and rebuild")
        print(f"Loaded {split} cache: {n:,} sequences ({directory})", flush=True)
        return directory, meta

    directory.mkdir(parents=True, exist_ok=True)
    # Temporary files prevent an interrupted build from looking complete.
    tmp_x, tmp_y = directory / "x.f32.partial", directory / "y.f32.partial"
    offsets = [0]
    print(f"Building {split} sequence cache ({len(ids):,} patients); first build reads CSV once", flush=True)
    try:
        with tmp_x.open("wb") as fx, tmp_y.open("wb") as fy:
            for i, patient_id in enumerate(ids, 1):
                frame = pd.read_csv(patient_path(root, patient_id), sep="|")
                x, y = make_sequences(frame, stats)
                if x.shape[1:] != WINDOW_SHAPE or len(x) != len(y):
                    raise ValueError(f"Sequence shape mismatch: {patient_id}")
                np.asarray(x, dtype=np.float32).tofile(fx)
                np.asarray(y, dtype=np.float32).tofile(fy)
                offsets.append(offsets[-1] + len(y))
                if i % 2000 == 0:
                    print(f"Cache {split}: {i:,}/{len(ids):,} patients", flush=True)
        if not offsets[-1]:
            raise ValueError(f"No {split} sequences to cache")
        os.replace(tmp_x, x_path)
        os.replace(tmp_y, y_path)
        meta = {"version": CACHE_VERSION, "digest": digest, "split": split,
                "samples": offsets[-1], "offsets": offsets,
                "patients": len(ids), "shape": list(WINDOW_SHAPE)}
        # Metadata is written last, marking a complete cache.
        tmp_meta = directory / "meta.json.partial"
        tmp_meta.write_text(json.dumps(meta), encoding="utf-8")
        os.replace(tmp_meta, metadata)
        print(f"Cached {split}: {offsets[-1]:,} sequences ({directory})", flush=True)
        return directory, meta
    except BaseException:
        tmp_x.unlink(missing_ok=True)
        tmp_y.unlink(missing_ok=True)
        raise


class CachedPatientSequences:
    """Same patient/within-patient ordering and batching as PatientSequences."""

    def __init__(self, ids, root, stats, batch_size, shuffle, seed, cache_dir, split):
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.epoch = 0
        self.directory, meta = load_or_build(cache_dir, split, ids, root, stats)
        self.offsets = meta["offsets"]
        self.n = meta["samples"]
        self.n_patients = len(ids)

    def __iter__(self):
        x_data = np.memmap(self.directory / "x.f32", dtype=np.float32,
                           mode="r", shape=(self.n, *WINDOW_SHAPE))
        y_data = np.memmap(self.directory / "y.f32", dtype=np.float32,
                           mode="r", shape=(self.n,))
        rng = np.random.default_rng(self.seed + self.epoch)
        order = list(range(self.n_patients))
        if self.shuffle:
            rng.shuffle(order)
        bx, by = [], []
        for i in order:
            start, stop = self.offsets[i], self.offsets[i + 1]
            if start == stop:
                continue
            indices = np.arange(stop - start)
            if self.shuffle:
                # Preserve the original loader's per-patient permutation calls.
                indices = rng.permutation(stop - start)
            for index in indices:
                bx.append(x_data[start + index])
                by.append(y_data[start + index])
                if len(by) >= self.batch_size:
                    yield (torch.from_numpy(np.stack(bx)),
                           torch.from_numpy(np.asarray(by, dtype=np.float32)))
                    bx, by = [], []
        if by:
            yield (torch.from_numpy(np.stack(bx)),
                   torch.from_numpy(np.asarray(by, dtype=np.float32)))
