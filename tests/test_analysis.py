import numpy as np

from litm.analysis import relative_doc_attention, score_answer, wilson_ci


def test_score_answer_first_line_only():
    assert score_answer("The Wilhelm Conrad Röntgen.\nmore", ["Wilhelm Conrad Röntgen"]) == 1
    assert score_answer("I don't know\nWilhelm Conrad Röntgen", ["Wilhelm Conrad Röntgen"]) == 0
    assert score_answer("an apple", ["The Apple"]) == 1


def test_wilson_ci():
    lo, hi = wilson_ci(150, 300)
    assert abs(lo - 0.4438) < 1e-3 and abs(hi - 0.5562) < 1e-3
    assert wilson_ci(0, 10)[0] == 0.0


SPANS = [(2, 4), (4, 8), (8, 12)]


def test_relative_attention_uniform_is_one():
    rows = np.full((3, 12), 1 / 12)
    assert np.allclose(relative_doc_attention(rows, SPANS), 1.0)


def test_relative_attention_ignores_sink():
    rows = np.full((1, 12), 0.5 / 10)
    rows[0, :2] = 0.25  # sink nos 2 primeiros tokens
    assert np.allclose(relative_doc_attention(rows, SPANS), 1.0)


def test_relative_attention_concentrated():
    rows = np.zeros((1, 12))
    rows[0, 5] = 1.0
    # densidade (1/4) / média (1/10): os spans cobrem 10 tokens
    assert np.allclose(relative_doc_attention(rows, SPANS), [[0.0, 2.5, 0.0]])
