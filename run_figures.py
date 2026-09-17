"""Regenerate the monograph's two chart figures from the stored artifacts.

These charts were produced once, by hand, and then embedded in the .docx. That
made them the only numbers in the work with no path back to the data: when the
artifacts were regenerated, Figura 6 silently disagreed with Tabela 11 and
nothing detected it. This script closes that gap -- both figures are now a
function of `data/results/`, and re-running it after any experiment keeps them
honest.

Usage:
    python run_figures.py                 # writes to data/results/figuras/
    python run_figures.py --embed         # also swaps them into the .docx

The greyscale palette, the legend above the axes and the comma decimal
separator reproduce the style of the embedded originals, so a regenerated
figure drops into the document without a visible change of house style.
"""

import argparse
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ESCURO, MEDIO, CLARO = "#2b2b2b", "#808080", "#d9d9d9"
ALGORITMOS = [
    ("random_forest", "Floresta aleatória", ESCURO, "o"),
    ("svm", "Máquina de vetores de suporte", MEDIO, "s"),
    ("logistic_regression", "Regressão logística", CLARO, "^"),
]
CONFIGS = ["full", "sentinel_fixed", "no_leak", "no_leak_strict"]

# The .docx embeds figures as media parts; these are the ones the captions
# Figura 6 and Figura 7 point at.
DOCX = Path("docs/monography/TCC - Artur Rios da Silva Oliveira.docx")
MEDIA = {"figura6_ablacao_exact.png": "word/media/image2.png",
         "figura7_recall_por_semente.png": "word/media/image3.png"}

_virgula = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))


def _estilo(ax) -> None:
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_linewidth(1.6)
        ax.spines[lado].set_color("black")
    ax.tick_params(labelsize=17, colors="black", width=1.4, length=6)
    ax.yaxis.set_major_formatter(_virgula)


def figura6(destino: Path) -> None:
    """Ablation F1-macro by feature configuration, exact scenario."""
    dados = pd.read_csv("data/results/ablation/ablation_summary.csv")
    dados = dados[dados.scenario == "exact"]

    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    largura = 0.26
    posicoes = range(len(CONFIGS))

    for i, (chave, rotulo, cor, _) in enumerate(ALGORITMOS):
        valores = [
            dados[(dados.config == c) & (dados.algorithm == chave)].f1_macro.iloc[0]
            for c in CONFIGS
        ]
        ax.bar([p + (i - 1) * largura for p in posicoes], valores, largura,
               label=rotulo, color=cor, edgecolor="black", linewidth=1.2)

    ax.set_xticks(list(posicoes))
    ax.set_xticklabels(CONFIGS)
    ax.set_xlabel("Configuração de atributos", fontsize=18, color="black", labelpad=10)
    ax.set_ylabel("F1-macro", fontsize=18, color="black", labelpad=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _estilo(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=17, handlelength=1.6, columnspacing=2.2)
    fig.savefig(destino, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figura7(destino: Path) -> None:
    """Exception recall per seed, under the precision floor."""
    dados = pd.read_csv("data/results/comparison/per_seed_metrics.csv")

    fig, ax = plt.subplots(figsize=(12.5, 6.3))
    for chave, rotulo, cor, marcador in ALGORITMOS:
        serie = dados[dados.algorithm == chave].sort_values("seed")
        ax.plot(serie.seed, serie.recall_excecao, marker=marcador, label=rotulo,
                color=cor, markerfacecolor=cor, markeredgecolor="black",
                markersize=11, linewidth=2.2)

    ax.set_xticks(sorted(dados.seed.unique()))
    ax.set_xlabel("Semente aleatória", fontsize=18, color="black", labelpad=10)
    ax.set_ylabel("Recall da classe de exceção", fontsize=18, color="black", labelpad=10)
    ax.set_ylim(0.70, 0.87)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85])
    _estilo(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              frameon=False, fontsize=17, handlelength=1.8, columnspacing=2.2)
    fig.savefig(destino, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def embutir(pasta: Path) -> None:
    """Replace the media parts inside the .docx, leaving everything else alone.

    A .docx is a zip, and python-docx offers no way to swap an image part in
    place, so the archive is rewritten entry by entry with the two pictures
    substituted. Every other part is copied byte for byte.
    """
    original = DOCX.with_suffix(".docx.bak")
    shutil.copy2(DOCX, original)

    origem = zipfile.ZipFile(original)
    with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as saida:
        for item in origem.infolist():
            substituta = next((n for n, alvo in MEDIA.items() if alvo == item.filename), None)
            if substituta:
                saida.writestr(item, (pasta / substituta).read_bytes())
                print(f"  substituída {item.filename} <- {substituta}")
            else:
                saida.writestr(item, origem.read(item.filename))
    origem.close()
    original.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embed", action="store_true",
                        help="também substituir as figuras dentro do .docx da monografia")
    parser.add_argument("--output", default="data/results/figuras")
    args = parser.parse_args()

    pasta = Path(args.output)
    pasta.mkdir(parents=True, exist_ok=True)

    figura6(pasta / "figura6_ablacao_exact.png")
    figura7(pasta / "figura7_recall_por_semente.png")
    print(f"figuras escritas em {pasta}/")

    if args.embed:
        embutir(pasta)
        print(f"figuras embutidas em {DOCX}")


if __name__ == "__main__":
    main()
