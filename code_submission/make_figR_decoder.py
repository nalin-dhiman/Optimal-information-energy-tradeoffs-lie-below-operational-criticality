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

CSV = TBL_DIR / "decoder_robustness.csv"
JC = 0.60


def make_figure():
    d = pd.read_csv(CSV).sort_values("J")

    # Find J* for each decoder
    j_star_ridge = d.loc[d["I_dec_Ridge"].idxmax(), "J"]
    j_star_ols = d.loc[d["I_dec_OLS"].idxmax(), "J"]
    j_star = j_star_ridge  # identical

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))


    ax.plot(d["J"], d["I_dec_Ridge"], "k-o", markersize=6, linewidth=2,
            label="Ridge decoder")


    ax.plot(d["J"], d["I_dec_OLS"], color="darkorange", linestyle="--",
            marker="s", markersize=6, linewidth=2,
            label="OLS autoregressive decoder")


    ax.axvline(j_star, color="blue", linestyle="-", linewidth=1.5,
               label=rf"$J^* = {j_star:.2f}$")


    ax.axvline(JC, color="red", linestyle="--", linewidth=1.5,
               label=rf"$J_c \approx {JC:.2f}$")

    ax.set_xlabel(r"Coupling $J$")
    ax.set_ylabel("Objective")
    ax.margins(y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_decoder_robustness_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
