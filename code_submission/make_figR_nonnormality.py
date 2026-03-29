#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TBL_DIR = Path(
    
)
FIG_DIR = Path(
    
)

CSV = TBL_DIR / "nonnormality_summary.csv"
JC = 0.60
N_TARGET = 5000


def make_figure():
    df = pd.read_csv(CSV)
    d = df[df["N"] == N_TARGET].sort_values("J")

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))


    ax = axes[0, 0]
    ax.plot(d["J"], d["spectral_abscissa"], "k-o", markersize=6, linewidth=1.8,
            label=r"Spectral abscissa $\alpha(M)$")
    ax.plot(d["J"], d["numerical_abscissa"], "r--s", markersize=6, linewidth=1.8,
            label=r"Numerical abscissa $\omega(M)$")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.axvline(JC, color="red", linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_ylabel("Abscissa")
    ax.set_xlabel(r"Coupling $J$")
    ax.set_title("(a) Spectral and numerical abscissas")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax = axes[0, 1]
    ax.plot(d["J"], d["henrici_departure"], "o-", color="tab:blue", markersize=6, linewidth=1.8)
    ax.axvline(JC, color="red", linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_ylabel("Henrici departure")
    ax.set_xlabel(r"Coupling $J$")
    ax.set_title("(b) Departure from normality")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax = axes[1, 0]
    ax.plot(d["J"], d["kappa_P"], "o-", color="tab:green", markersize=6, linewidth=1.8)
    ax.axvline(JC, color="red", linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_ylabel(r"Eigenvector condition number $\kappa(P)$")
    ax.set_xlabel(r"Coupling $J$")
    ax.set_title(r"(c) Eigenvector conditioning")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax = axes[1, 1]
    ax.plot(d["J"], d["mean_rate"], "k-^", markersize=6, linewidth=1.8)
    ax.axvline(JC, color="red", linestyle="--", linewidth=1.3, alpha=0.6)
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_xlabel(r"Coupling $J$")
    ax.set_title("(d) Mean rate")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h, l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(h_pad=2.5, w_pad=2.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_nonnormality_vs_J_N5000_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
