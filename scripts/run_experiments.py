"""Roda os experimentos e grava em results/.

Exemplos:
  python scripts/run_experiments.py --model qwen3-4b --exp accuracy --n 300
  python scripts/run_experiments.py --model qwen3-4b --exp attention --n 100
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litm.analysis import relative_doc_attention, score_answer  # noqa: E402
from litm.data import build_prompt, download, load_examples, move_gold, sample_ids  # noqa: E402
from litm.model import chat_text, doc_token_spans, generate, last_row_attention, load  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--exp", choices=["accuracy", "attention"], required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--positions", default=None, help="1-based, ex.: 1,5,10,15,20")
    p.add_argument("--out", default="results")
    p.add_argument("--data", default=None)
    a = p.parse_args(argv)

    default = "1,5,10,15,20" if a.exp == "accuracy" else ",".join(map(str, range(1, 21)))
    positions = [int(x) for x in (a.positions or default).split(",")]
    examples = load_examples(Path(a.data) if a.data else download())
    qids = sample_ids(len(examples), a.n)
    model, tok = load(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if a.exp == "accuracy":
        rows = []
        for qid in tqdm(qids, desc=f"{a.model} accuracy"):
            ex = examples[qid]
            for pos in positions:
                text = chat_text(tok, build_prompt(ex["question"], move_gold(ex, pos))[0])
                ans = generate(model, tok, text)
                rows.append((a.model, pos, qid, score_answer(ans, ex["answers"]), ans.split("\n")[0].strip()))
        pd.DataFrame(rows, columns=["model", "position", "qid", "correct", "answer"]).to_csv(
            out / f"accuracy_{a.model}.csv", index=False)
    else:
        rel = None
        for j, qid in enumerate(tqdm(qids, desc=f"{a.model} attention")):
            ex = examples[qid]
            for i, pos in enumerate(positions):
                prompt, docs = build_prompt(ex["question"], move_gold(ex, pos))
                text = chat_text(tok, prompt)
                r = relative_doc_attention(last_row_attention(model, tok, text), doc_token_spans(tok, text, docs))
                if rel is None:
                    rel = np.zeros((len(positions), len(qids), *r.shape), dtype=np.float16)
                rel[i, j] = r
        np.savez_compressed(out / f"attention_{a.model}.npz", rel=rel, qids=np.array(qids),
                            positions=np.array(positions))

    mem = f", pico VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""
    print(f"{a.model} {a.exp}: {time.time() - t0:.0f}s{mem}")


if __name__ == "__main__":
    main()
