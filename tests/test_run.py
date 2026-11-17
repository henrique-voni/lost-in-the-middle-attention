import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_experiments import main  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "gold_at_0_head3.jsonl"
ARGS = ["--model", "tiny", "--data", str(FIX), "--n", "2", "--positions", "1,20"]


def test_smoke_accuracy(tmp_path):
    main(ARGS + ["--exp", "accuracy", "--out", str(tmp_path)])
    df = pd.read_csv(tmp_path / "accuracy_tiny.csv")
    assert len(df) == 4
    assert list(df.columns) == ["model", "position", "qid", "correct", "answer"]
    assert set(df.correct) <= {0, 1}


def test_smoke_attention(tmp_path):
    main(ARGS + ["--exp", "attention", "--out", str(tmp_path)])
    z = np.load(tmp_path / "attention_tiny.npz")
    assert z["rel"].shape == (2, 2, 30, 20)  # SmolLM2-135M tem 30 camadas
    assert np.isfinite(z["rel"]).all()
    assert list(z["positions"]) == [1, 20]
