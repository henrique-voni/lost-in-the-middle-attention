from pathlib import Path

from litm.data import build_prompt, load_examples, move_gold, sample_ids

FIX = Path(__file__).parent / "fixtures"


def test_move_gold_matches_paper_file():
    ex0 = load_examples(FIX / "gold_at_0_head3.jsonl")
    ex9 = load_examples(FIX / "gold_at_9_head3.jsonl")
    for a, b in zip(ex0, ex9):
        ours = [(c["title"], c["text"]) for c in move_gold(a, 10)]
        assert ours == [(c["title"], c["text"]) for c in b["ctxs"]]
        assert move_gold(a, 10)[9]["isgold"]


def test_move_gold_does_not_mutate():
    ex = load_examples(FIX / "gold_at_0_head3.jsonl")[0]
    before = [c["title"] for c in ex["ctxs"]]
    move_gold(ex, 20)
    assert [c["title"] for c in ex["ctxs"]] == before


def test_sample_ids_prefix():
    assert sample_ids(2655, 100) == sample_ids(2655, 300)[:100]
    assert len(set(sample_ids(2655, 300))) == 300


def test_build_prompt_format():
    prompt, docs = build_prompt("q?", [{"title": "A", "text": "x"}, {"title": "B", "text": "y"}])
    assert docs == ["Document [1](Title: A) x", "Document [2](Title: B) y"]
    assert prompt == (
        "Write a high-quality answer for the given question using only the provided "
        "search results (some of which might be irrelevant).\n\n"
        "Document [1](Title: A) x\nDocument [2](Title: B) y\n\nQuestion: q?\nAnswer:"
    )
