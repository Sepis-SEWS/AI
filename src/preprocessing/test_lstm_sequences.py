"""Synthetic tests for train-only LSTM preprocessing."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lstm_sequences import FEATURES, fit_train_stats, make_sequences, read_manifest


def patient(hr, length=25):
    d = pd.DataFrame({
        "ICULOS": range(1, length+1),
        "SepsisLabel": [0]*(length-1)+[1],
        **{f: [1.0]*length for f in FEATURES},
    })
    d["HR"] = hr
    d["Lactate"] = [np.nan]*length
    return d


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for group in ("training_setA", "training_setB"):
            (root/group).mkdir()
        for name, hr in (("train", [80.0]*25), ("validation", [999.0]*25),
                         ("test", [500.0]*25)):
            patient(hr).to_csv(root/"training_setA"/f"{name}.psv", sep="|", index=False)
        # Fit requires at least one finite observation for each feature.
        frame = patient([80.0]*25)
        frame["Lactate"] = [2.0]*25
        frame.to_csv(root/"training_setA"/"train.psv", sep="|", index=False)
        stats = fit_train_stats(["training_setA/train"], root)
        assert stats["median"]["HR"] == 80.0
        assert stats["median"]["Lactate"] == 2.0
        # Changing validation must not affect the training statistics.
        stats_again = fit_train_stats(["training_setA/train"], root)
        assert stats == stats_again
        eval_patient = patient([np.nan] + [999.0]*24)
        x, y = make_sequences(eval_patient, stats)
        assert x.shape == (2,24,64), x.shape
        assert y.tolist() == [0,1]
        assert np.isfinite(x).all()
        assert x[0,0,0] == 0.0  # initial HR missing -> train median 80
        assert x[0,0,1] == 0.0  # HR observed mask
        assert x[0,0,3] == 1.0  # HR never observed
        assert x[0,0,4] == 0.0  # Temp observed value: train median 1
        assert x[0,0,29] == 0.0  # Lactate observed mask, no observed labs
        assert x[0,0,31] == 1.0  # Lactate never-observed channel
        short, labels = make_sequences(eval_patient.iloc[:20], stats)
        assert short.shape == (0,24,64) and len(labels)==0
        empty = root/"empty.json"
        empty.write_text("")
        try:
            read_manifest(empty)
            raise AssertionError("Expected empty split rejection")
        except ValueError:
            pass
    print("PASS: train-only statistics, finite 24x64 LSTM inputs, labels, short patients")


if __name__ == "__main__":
    main()
