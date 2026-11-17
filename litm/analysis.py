"""Métrica do paper, intervalo de confiança e agregação da atenção por documento."""
import math
import re
import string

import numpy as np


def normalize_answer(s: str) -> str:
    """Normalização do SQuAD, como em lost_in_the_middle/metrics.py."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_answer(prediction: str, answers: list[str]) -> int:
    """best_subspan_em sobre a 1ª linha da resposta, como o avaliador do paper
    (evita 'acertar' copiando o contexto, que sempre contém a resposta)."""
    pred = normalize_answer(prediction.split("\n")[0].strip())
    return int(any(normalize_answer(a) in pred for a in answers))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def relative_doc_attention(rows: np.ndarray, spans: list[tuple[int, int]]) -> np.ndarray:
    """(L, T) -> (L, D): densidade de atenção de cada documento relativa à média dos documentos.

    1.0 = o documento recebe atenção proporcional ao seu tamanho. Tokens fora dos spans
    (BOS/attention sink, instrução, pergunta) ficam de fora por construção.
    """
    mass = np.stack([rows[:, s:e].sum(-1) for s, e in spans], axis=-1)
    lens = np.array([e - s for s, e in spans], dtype=float)
    overall = mass.sum(-1, keepdims=True) / lens.sum()
    return (mass / lens) / overall
