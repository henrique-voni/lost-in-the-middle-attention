import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from litm.data import build_prompt, load_examples
from litm.model import MODELS, chat_text, doc_token_spans, generate, last_row_attention, load

FIX = Path(__file__).parent / "fixtures"
EX = load_examples(FIX / "gold_at_0_head3.jsonl")[0]
CTXS3 = EX["ctxs"][:3]


@pytest.fixture(scope="module")
def tiny():
    m, tok = load("tiny", device="cpu")
    text = chat_text(tok, build_prompt(EX["question"], CTXS3)[0])
    return m, tok, text


def _eager_rows(model, tok, text):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    model.set_attn_implementation("eager")
    try:
        with torch.no_grad():
            att = model(**ids, output_attentions=True).attentions
    finally:
        model.set_attn_implementation("last_row_sdpa")
    return torch.stack([a[0, :, -1, :].float().mean(0) for a in att]).cpu().numpy()


def test_last_row_matches_eager(tiny):
    m, tok, text = tiny
    assert np.allclose(last_row_attention(m, tok, text), _eager_rows(m, tok, text), atol=1e-5)


def test_logits_unchanged(tiny):
    m, tok, text = tiny
    ref = AutoModelForCausalLM.from_pretrained(MODELS["tiny"], attn_implementation="sdpa", dtype=torch.float32)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        assert torch.allclose(m(**ids).logits, ref(**ids).logits, atol=1e-5)


def test_rows_sum_to_one(tiny):
    m, tok, text = tiny
    assert np.allclose(last_row_attention(m, tok, text).sum(-1), 1.0, atol=1e-4)


def test_doc_token_spans_roundtrip(tiny):
    _, tok, _ = tiny
    prompt, docs = build_prompt("who?", CTXS3)
    text = chat_text(tok, prompt)
    spans = doc_token_spans(tok, text, docs)
    assert all(a < b <= c < d for (a, b), (c, d) in zip(spans, spans[1:]))
    ids = tok(text, add_special_tokens=False).input_ids
    for (s, e), c in zip(spans, CTXS3):
        assert f"Title: {c['title']}" in tok.decode(ids[s:e])


def test_doc_token_spans_duplicate_docs(tiny):
    _, tok, _ = tiny
    prompt, docs = build_prompt("who?", [CTXS3[0], CTXS3[0]])
    spans = doc_token_spans(tok, chat_text(tok, prompt), docs)
    assert spans[0] != spans[1]


def test_generate_stops_at_first_newline_same_first_line(tiny):
    # a métrica só usa a 1ª linha; com greedy, parar no "\n" não muda o resultado, só o tempo
    m, tok, _ = tiny
    text = chat_text(tok, "List five fruits, one per line.")
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        full = m.generate(**ids, max_new_tokens=100, do_sample=False, temperature=None, top_p=None, top_k=None)
    full = tok.decode(full[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
    assert "\n" in full.strip("\n"), "pré-condição: o modelo gera mais de uma linha neste prompt"
    out = generate(m, tok, text)
    assert out.rstrip("\n").count("\n") == 0
    assert out.split("\n")[0] == full.split("\n")[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="precisa de CUDA")
def test_load_uses_fp16_on_cuda():
    # bf16 deriva do fp32 em até 0,05 na atenção relativa (Qwen3); fp16 fica em 0,001
    m, _ = load("tiny")
    assert m.dtype == torch.float16


@pytest.mark.parametrize("key", ["smollm3-3b", "qwen3-4b"])
def test_chat_text_no_open_think(key):
    t = chat_text(AutoTokenizer.from_pretrained(MODELS[key]), "hi")
    assert t.count("<think>") == t.count("</think>")


@pytest.mark.skipif(not os.environ.get("LITM_GPU"), reason="GPU, modelos reais")
@pytest.mark.parametrize("key", ["qwen3-4b", "phi4-mini", "smollm3-3b"])
def test_last_row_matches_eager_real(key):
    m, tok = load(key)
    text = chat_text(tok, build_prompt(EX["question"], EX["ctxs"][:5])[0])
    assert np.allclose(last_row_attention(m, tok, text), _eager_rows(m, tok, text), atol=5e-3)
    del m
    torch.cuda.empty_cache()
