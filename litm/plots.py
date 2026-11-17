"""Figuras do experimento e animação do post."""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from litm.analysis import wilson_ci  # noqa: E402

# Paleta validada (dataviz): slots categóricos 1-3 em ordem fixa, cor segue o modelo.
NAMES = {"qwen3-4b": "Qwen3-4B", "phi4-mini": "Phi-4-mini", "smollm3-3b": "SmolLM3-3B"}
COLORS = {"qwen3-4b": "#2a78d6", "phi4-mini": "#eb6834", "smollm3-3b": "#1baf7a"}
FALLBACK = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
}
CAPTION = ("Dados: Liu et al. (2023), NQ-Open, 20 documentos. Atenção = último token do prompt, "
           "média de cabeças e camadas, relativa ao tamanho do documento.")


def _name(m):
    return NAMES.get(m, m)


def _color(m, i):
    return COLORS.get(m, FALLBACK[i % len(FALLBACK)])


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "position"]).correct.agg(["sum", "count"]).reset_index()
    ci = [wilson_ci(int(k), int(n)) for k, n in zip(g["sum"], g["count"])]
    return pd.DataFrame({"model": g.model, "position": g.position, "acc": g["sum"] / g["count"],
                         "lo": [c[0] for c in ci], "hi": [c[1] for c in ci]})


def gold_curve(rel: np.ndarray, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Atenção relativa no ouro por posição: média ± 1,96·SE sobre as perguntas."""
    g = np.stack([rel[i, :, :, p - 1].astype(float).mean(-1) for i, p in enumerate(positions)])  # (P, N)
    return g.mean(1), 1.96 * g.std(1, ddof=1) / np.sqrt(g.shape[1])


def positional_bias(rel: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Atenção média em cada posição só quando ela NÃO tem o ouro (viés puro de posição)."""
    positions = np.asarray(positions)
    return np.array([rel[positions != d + 1][..., d].astype(float).mean() for d in range(rel.shape[-1])])


def _direct_labels(ax, ends, min_gap):
    """Rótulo com o nome no fim de cada linha, afastando os que colidem."""
    ends = sorted(ends, key=lambda e: e[1])
    ys = []
    for _, y, _ in ends:
        ys.append(max(y, ys[-1] + min_gap) if ys else y)
    for (m, _, _c), y in zip(ends, ys):
        ax.annotate(_name(m), (20, y), xytext=(12, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=9,
                    annotation_clip=False)


def plot_main(tables: dict, curves: dict, path: Path) -> None:
    with plt.rc_context(STYLE):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True, constrained_layout=True)
        ends1, ends2 = [], []
        for i, m in enumerate(tables):
            t, c = tables[m], _color(m, i)
            dodge = 0.25 * (i - (len(tables) - 1) / 2)  # separa as barras de erro na mesma posição
            a1.errorbar(t.position + dodge, 100 * t.acc, yerr=[100 * (t.acc - t.lo), 100 * (t.hi - t.acc)],
                        color=c, lw=2, marker="o", ms=7, capsize=3, label=_name(m))
            ends1.append((m, 100 * t.acc.iloc[-1], c))
        for i, m in enumerate(curves):
            mean, ci = curves[m]
            assert len(mean) == 20, "o Experimento 2 cobre as 20 posições"
            x = np.arange(1, 21)
            a2.fill_between(x, mean - ci, mean + ci, color=_color(m, i), alpha=0.15, lw=0)
            a2.plot(x, mean, color=_color(m, i), lw=2, marker="o", ms=4, label=_name(m))
            ends2.append((m, mean[-1], _color(m, i)))
        a2.axhline(1.0, color=INK2, lw=1, ls="--")
        a2.annotate("atenção uniforme", (1, 1.0), xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
        _direct_labels(a1, ends1, 3.0)
        _direct_labels(a2, ends2, 0.25)
        a1.set_ylabel("Acurácia (%)")
        a1.set_title("Acurácia × posição do documento relevante (IC 95%)", loc="left", fontsize=11)
        a2.set_ylabel("Atenção relativa no documento relevante")
        a2.set_title("Atenção no documento relevante (1 = proporcional ao tamanho)", loc="left", fontsize=11)
        a2.set_xlabel("Posição do documento relevante entre 20")
        a2.set_xticks([1, 5, 10, 15, 20])
        a1.legend(loc="lower left", frameon=False, ncol=3, fontsize=9)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def plot_layer_heatmap(rel: np.ndarray, positions: np.ndarray, model: str, path: Path) -> None:
    mat = np.stack([rel[i, :, :, p - 1].astype(float).mean(0) for i, p in enumerate(positions)], axis=1)  # (L, P)
    with plt.rc_context({**STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(7, 5.5), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap=BLUES,
                       extent=(positions[0] - 0.5, positions[-1] + 0.5, -0.5, mat.shape[0] - 0.5))
        fig.colorbar(im, ax=ax, label="Atenção relativa no documento relevante")
        ax.set_xlabel("Posição do documento relevante entre 20")
        ax.set_ylabel("Camada")
        ax.set_title(f"{_name(model)}: atenção no documento relevante por camada", loc="left", fontsize=11)
        fig.savefig(path, dpi=200)
        plt.close(fig)


def make_gif(rels: dict, tables: dict, path: Path, mp4: bool = True) -> None:
    """Animação 4:5 (1080×1350): o documento relevante desliza de 1 a 20."""
    models = list(rels)
    positions = rels[models[0]][1]
    bars = {m: np.stack([r[i].astype(float).mean((0, 1)) for i in range(len(p))]) for m, (r, p) in rels.items()}
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(5.4, 6.75))
        gs = fig.add_gridspec(len(models) + 1, 1, height_ratios=[1] * len(models) + [1.5],
                              left=0.12, right=0.96, top=0.86, bottom=0.12, hspace=0.55)
        axes = [fig.add_subplot(gs[i]) for i in range(len(models))]
        acc_ax = fig.add_subplot(gs[-1])
        title = fig.text(0.04, 0.955, "", fontsize=14, weight="bold")
        fig.text(0.04, 0.915, "Lost in the Middle: quanta atenção cada um dos 20 documentos recebe",
                 fontsize=9.5, color=INK2)
        fig.text(0.04, 0.025, CAPTION, fontsize=6.5, color=INK2, wrap=True)

        def draw(k):
            p = positions[k]
            title.set_text(f"Documento relevante na posição {p}")
            for i, (ax, m) in enumerate(zip(axes, models)):
                ax.clear()
                colors = [_color(m, i) if d + 1 == p else NEUTRAL for d in range(20)]
                ax.bar(np.arange(1, 21), bars[m][k], color=colors, width=0.8)
                ax.axhline(1.0, color=INK2, lw=0.8, ls="--")
                ax.set_ylim(0, bars[m].max() * 1.1)
                ax.set_xlim(0.4, 20.6)
                ax.set_xticks([1, 5, 10, 15, 20])
                ax.tick_params(labelsize=8)
                ax.set_title(_name(m), loc="left", fontsize=10, color=INK)
                ax.grid(axis="x", visible=False)
            acc_ax.clear()
            for i, m in enumerate(tables):
                t = tables[m][tables[m].position <= p]
                acc_ax.plot(t.position, 100 * t.acc, color=_color(m, i), lw=2, marker="o", ms=6, label=_name(m))
            acc_ax.axvline(p, color=INK2, lw=0.8, alpha=0.5)
            lo = min(tb.acc.min() for tb in tables.values())
            hi = max(tb.acc.max() for tb in tables.values())
            acc_ax.set_ylim(100 * lo - 8, 100 * hi + 8)
            acc_ax.set_xlim(0.4, 20.6)
            acc_ax.set_xticks([1, 5, 10, 15, 20])
            acc_ax.tick_params(labelsize=8)
            acc_ax.set_title("Acurácia (%)", loc="left", fontsize=10)
            acc_ax.set_xlabel("Posição do documento relevante", fontsize=9)
            acc_ax.legend(loc="lower right", bbox_to_anchor=(1, 1), frameon=False, fontsize=8,
                          ncol=len(tables), handlelength=1.2, borderaxespad=0.1)

        frames = list(range(len(positions))) + [len(positions) - 1] * 6  # segura o último quadro
        anim = FuncAnimation(fig, draw, frames=frames)
        anim.save(path, writer=PillowWriter(fps=2), dpi=200)
        if mp4 and shutil.which("ffmpeg"):
            anim.save(Path(path).with_suffix(".mp4"), writer=FFMpegWriter(fps=2), dpi=200)
        plt.close(fig)
