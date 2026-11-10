"""AI-9 inference contract tests. Synthetic data only; no Test reads or model scoring."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "inference"))
from predict_lstm import FEATURES, build_inference_frame, predict  # noqa: E402


def observations(n: int) -> list[dict]:
    return [{"ICULOS": t, **{f: (80.0 if f == "HR" and t == 1 else None)
                             for f in FEATURES}} for t in range(1, n + 1)]


def synthetic_stats() -> dict:
    return {
        "features": FEATURES, "window_rows": 24, "age_cap_hours": 48.0,
        "median": {f: 1.0 for f in FEATURES},
        "scale_iqr": {f: 2.0 for f in FEATURES},
    }


class ConstantLogit(torch.nn.Module):
    def __init__(self, score: float):
        super().__init__()
        self.register_buffer("logit", torch.tensor(np.log(score / (1 - score)),
                                                   dtype=torch.float32))

    def forward(self, x):
        if x.shape[1:] != (24, 64):
            raise ValueError("Inference shape mismatch")
        return self.logit.repeat(len(x))


class InferenceContractTests(unittest.TestCase):
    def test_rejects_label_leakage(self):
        records = observations(24)
        records[0]["SepsisLabel"] = 0
        with self.assertRaisesRegex(ValueError, "SepsisLabel"):
            build_inference_frame(records)

    def test_rejects_bad_time(self):
        records = observations(24)
        records[-1]["ICULOS"] = 23
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_inference_frame(records)

    def test_rejects_missing_feature(self):
        records = observations(24)
        for record in records:
            del record["HR"]
        with self.assertRaisesRegex(ValueError, "Missing required"):
            build_inference_frame(records)

    def test_insufficient_history_has_no_score(self):
        result = predict(observations(23), ConstantLogit(0.8), synthetic_stats())
        self.assertEqual(result["status"], "insufficient_history")
        self.assertIsNone(result["risk_score"])
        self.assertFalse(result["alert"])

    def test_threshold_without_calibration(self):
        stats = synthetic_stats()
        for score, expected_alert in ((0.1, False), (0.8, True)):
            result = predict(observations(24), ConstantLogit(score), stats)
            self.assertEqual(result["status"], "scored")
            self.assertAlmostEqual(result["risk_score"], score, places=6)
            self.assertEqual(result["alert"], expected_alert)
            self.assertIsNone(result["calibrated_probability"])

    def test_full_history_changes_age_feature(self):
        from lstm_sequences import make_sequences

        stats = synthetic_stats()
        full = observations(25)
        a, _ = make_sequences(build_inference_frame(full), stats)
        b, _ = make_sequences(build_inference_frame(full[1:]), stats)
        self.assertEqual(a.shape[1:], (24, 64))
        self.assertEqual(b.shape[1:], (24, 64))
        # HR observed in row 1: last 24-only history cannot recover observation age.
        self.assertNotEqual(float(a[-1, -1, 2]), float(b[-1, -1, 2]))


if __name__ == "__main__":
    unittest.main()
