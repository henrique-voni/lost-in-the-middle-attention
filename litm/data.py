"""Dados do paper Lost in the Middle (Liu et al., 2023) e montagem do prompt."""
import gzip
import json
import random
import urllib.request
from pathlib import Path

DATA_URL = (
    "https://github.com/nelson-liu/lost-in-the-middle/raw/main/qa_data/"
    "20_total_documents/nq-open-20_total_documents_gold_at_0.jsonl.gz"
)

# Template do paper (src/lost_in_the_middle/prompts/qa.prompt), verbatim.
QA_TEMPLATE = (
    "Write a high-quality answer for the given question using only the provided "
    "search results (some of which might be irrelevant).\n\n"
    "{search_results}\n\nQuestion: {question}\nAnswer:"
)


def download(dest: Path = Path("data")) -> Path:
    path = Path(dest) / DATA_URL.rsplit("/", 1)[1]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, path)
    return path


def load_examples(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def move_gold(example: dict, position: int) -> list[dict]:
    """Recoloca o documento-ouro na posição 1-based `position`.

    O script do paper faz `distratores.insert(gold_index, ouro)` na mesma lista,
    então tirar o ouro e reinserir reproduz exatamente os arquivos gold_at_k.
    """
    ctxs = list(example["ctxs"])
    (gold_idx,) = [i for i, c in enumerate(ctxs) if c["isgold"]]
    gold = ctxs.pop(gold_idx)
    ctxs.insert(position - 1, gold)
    return ctxs


def sample_ids(n_total: int, n: int, seed: int = 0) -> list[int]:
    # Prefixo de uma única permutação: N=100 é sempre subconjunto de N=300.
    return random.Random(seed).sample(range(n_total), n_total)[:n]


def build_prompt(question: str, ctxs: list[dict]) -> tuple[str, list[str]]:
    docs = [f"Document [{i + 1}](Title: {c['title']}) {c['text']}" for i, c in enumerate(ctxs)]
    return QA_TEMPLATE.format(search_results="\n".join(docs), question=question), docs
