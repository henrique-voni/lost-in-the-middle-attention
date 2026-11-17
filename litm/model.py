"""Carregamento dos modelos, spans de documentos e atenção do último token."""
import numpy as np
import torch
from transformers import AttentionInterface, AttentionMaskInterface, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    "phi4-mini": "microsoft/Phi-4-mini-instruct",
    "smollm3-3b": "HuggingFaceTB/SmolLM3-3B",
    "tiny": "HuggingFaceTB/SmolLM2-135M-Instruct",  # só testes/smoke
}

_ROWS: dict[int, torch.Tensor] = {}  # ponytail: estado global; batch=1, um forward por vez


def last_row_sdpa(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    """sdpa para a saída + pesos de atenção só da última query (q_último · Kᵀ).

    O eager materializa heads × T × T por camada (~3 GB com 4k tokens); aqui é heads × T.
    Com batch=1 e sem padding, a última linha da máscara causal enxerga todas as chaves.
    """
    window = kw.get("sliding_window")
    assert window is None or window >= key.shape[-2], "sliding window cortaria a última linha"
    out = sdpa_attention_forward(module, query, key, value, attention_mask,
                                 scaling=scaling, dropout=dropout, **kw)
    k = repeat_kv(key, module.num_key_value_groups)  # GQA
    s = scaling if scaling is not None else query.shape[-1] ** -0.5
    w = ((query[:, :, -1:, :] @ k.transpose(2, 3)) * s).float().softmax(-1)
    _ROWS[module.layer_idx] = w.mean(1)[0, 0].cpu()  # média das cabeças
    return out


AttentionInterface.register("last_row_sdpa", last_row_sdpa)
AttentionMaskInterface.register("last_row_sdpa", AttentionMaskInterface._global_mapping["sdpa"])


def load(key: str, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # fp16, não bf16: com bf16 a atenção do Qwen3 deriva até 0,05 do fp32 nas camadas
    # finais; com fp16 fica em ~0,001. Também é o que a T4 do Colab usa.
    dtype = torch.float32 if device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(MODELS[key])
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[key], dtype=dtype, attn_implementation="last_row_sdpa"
    ).to(device).eval()
    return model, tok


def chat_text(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def doc_token_spans(tokenizer, text: str, docs: list[str]) -> list[tuple[int, int]]:
    offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    spans, cursor = [], 0
    for doc in docs:
        start = text.find(doc, cursor)  # busca sequencial: documentos repetidos não colidem
        if start < 0:
            raise ValueError(f"documento não encontrado no prompt: {doc[:60]!r}")
        end = cursor = start + len(doc)
        idx = [i for i, (a, b) in enumerate(offsets) if start <= a and b <= end and b > a]
        spans.append((idx[0], idx[-1] + 1))
    return spans


@torch.no_grad()
def generate(model, tokenizer, text: str, max_new_tokens: int = 100) -> str:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    # Para no 1º "\n": a métrica do paper só usa a 1ª linha e, com greedy, ela não muda.
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         temperature=None, top_p=None, top_k=None,
                         stop_strings=["\n"], tokenizer=tokenizer)
    return tokenizer.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def last_row_attention(model, tokenizer, text: str) -> np.ndarray:
    """(L, T): atenção do último token do prompt (de onde sai o 1º token da resposta)."""
    _ROWS.clear()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    model(**ids)
    rows = torch.stack([_ROWS[i] for i in sorted(_ROWS)]).numpy()
    assert np.isfinite(rows).all(), "atenção com NaN/inf (fp16?)"
    return rows
