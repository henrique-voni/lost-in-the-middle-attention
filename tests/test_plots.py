import numpy as np
import pandas as pd

from litm.plots import (accuracy_table, gold_curve, make_gif, plot_layer_heatmap, plot_main,
                        positional_bias)

POS = np.arange(1, 21)


def _rel():
    rel = np.ones((20, 4, 3, 20))
    for i, p in enumerate(POS):
        rel[i, :, :, p - 1] = 2.0
    return rel


def test_gold_curve_constant():
    mean, ci = gold_curve(_rel(), POS)
    assert np.allclose(mean, 2.0) and np.allclose(ci, 0.0)


def test_positional_bias_excludes_gold():
    assert np.allclose(positional_bias(_rel(), POS), 1.0)


def test_accuracy_table():
    df = pd.DataFrame({"model": "m", "position": [1, 1, 20, 20], "qid": [0, 1, 0, 1], "correct": [1, 1, 0, 1]})
    t = accuracy_table(df)
    assert list(t.columns) == ["model", "position", "acc", "lo", "hi"]
    assert t.acc.tolist() == [1.0, 0.5]


def test_figures_smoke(tmp_path):
    df = pd.DataFrame({"model": "m", "position": [1, 10, 20] * 2, "qid": [0] * 3 + [1] * 3, "correct": [1, 0, 1, 1, 1, 0]})
    tables, rels = {"m": accuracy_table(df)}, {"m": (_rel(), POS)}
    plot_main(tables, {"m": gold_curve(_rel(), POS)}, tmp_path / "main.png")
    plot_layer_heatmap(_rel(), POS, "m", tmp_path / "heat.png")
    make_gif(rels, tables, tmp_path / "post.gif", mp4=False)
    for f in ["main.png", "heat.png", "post.gif"]:
        assert (tmp_path / f).stat().st_size > 0
