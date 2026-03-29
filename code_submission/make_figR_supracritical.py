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

CSV = TBL_DIR / "supracritical_probe.csv"
JC = 0.60


def make_figure():
    d = pd.read_csv(CSV).sort_values("J")


    j_star = d.loc[d["I_dec"].idxmax(), "J"]

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)


    ax_top.plot(d["J"], d["I_dec"], "k-o", markersize=6, linewidth=2,
                label="Decoder information")

    ax_top.axvline(j_star, color="blue", linestyle="-", linewidth=1.5,
                   label=rf"$J^* = {j_star:.3f}$")
    ax_top.axvline(JC, color="red", linestyle="--", linewidth=1.5,
                   label=rf"$J_c \approx {JC:.2f}$")

    ax_top.set_ylabel("Decoder information")
    ax_top.margins(y=0.08)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)


    ax_bot.plot(d["J"], d["mean_rate"], color="darkorange", linestyle="--",
                marker="s", markersize=6, linewidth=2,
                label="Mean firing rate")

    ax_bot.axvline(j_star, color="blue", linestyle="-", linewidth=1.5)
    ax_bot.axvline(JC, color="red", linestyle="--", linewidth=1.5)

    ax_bot.set_ylabel("Mean firing rate (Hz)")
    ax_bot.set_xlabel(r"Coupling $J$")
    ax_bot.margins(y=0.08)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)


    h_t, l_t = ax_top.get_legend_handles_labels()
    h_b, l_b = ax_bot.get_legend_handles_labels()
    fig.legend(
        h_t + h_b, l_t + l_b,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(h_pad=2.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_supracritical_probe_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
