import sys
from pathlib import Path

import numpy as np
import pandas as pd

from litm.data import sample_ids

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


def test_resume_accuracy_keeps_done_rows(tmp_path):
    # simula um run interrompido: 1 linha já gravada não pode ser recalculada nem duplicada
    qid = sample_ids(3, 2)[0]  # precisa ser uma pergunta da amostra
    pd.DataFrame([("tiny", 1, qid, 1, "SENTINEL")], columns=["model", "position", "qid", "correct", "answer"]).to_csv(
        tmp_path / "accuracy_tiny.csv", index=False)
    main(ARGS + ["--exp", "accuracy", "--out", str(tmp_path)])
    df = pd.read_csv(tmp_path / "accuracy_tiny.csv")
    assert len(df) == 4 and not df.duplicated(["qid", "position"]).any()
    assert (df.answer == "SENTINEL").sum() == 1


def test_resume_attention_continues_from_done(tmp_path):
    main(ARGS + ["--exp", "attention", "--out", str(tmp_path)])
    z = dict(np.load(tmp_path / "attention_tiny.npz"))
    z["rel"][:, 0], z["rel"][:, 1], z["done"] = 7.0, 0.0, 1  # 1ª pergunta feita, 2ª não
    np.savez(tmp_path / "attention_tiny.npz", **z)
    main(ARGS + ["--exp", "attention", "--out", str(tmp_path)])
    z = np.load(tmp_path / "attention_tiny.npz")
    assert (z["rel"][:, 0] == 7.0).all() and (z["rel"][:, 1] != 0).any() and int(z["done"]) == 2


def test_smoke_attention(tmp_path):
    main(ARGS + ["--exp", "attention", "--out", str(tmp_path)])
    z = np.load(tmp_path / "attention_tiny.npz")
    assert z["rel"].shape == (2, 2, 30, 20)  # SmolLM2-135M tem 30 camadas
    assert np.isfinite(z["rel"]).all()
    assert list(z["positions"]) == [1, 20]
