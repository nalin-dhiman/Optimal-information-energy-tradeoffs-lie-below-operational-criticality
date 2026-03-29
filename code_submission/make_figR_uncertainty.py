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
CSV = TBL_DIR / "uncertainty_main_figures.csv"

JC = 0.60  


def make_figure():
    df = pd.read_csv(CSV)
    df = df[df["beta_E"] == 0.0]

    sizes = sorted(df["N"].unique())  

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), sharey=True)
    axes_flat = axes.flatten()

    for i, N in enumerate(sizes):
        ax = axes_flat[i]
        sub = df[df["N"] == N].sort_values("J")


        ax.fill_between(
            sub["J"], sub["ci_lower_95"], sub["ci_upper_95"],
            color="black", alpha=0.12,
        )

        lbl_curve = "Mean objective" if i == 0 else ""
        ax.plot(
            sub["J"], sub["mean"],
            "k-o", markersize=5, linewidth=1.8,
            label=lbl_curve,
        )


        max_obj = sub["mean"].max()
        plateau = sub[sub["mean"] >= 0.95 * max_obj]
        if not plateau.empty:
            ax.axvspan(
                plateau["J"].min(), plateau["J"].max(),
                color="cornflowerblue", alpha=0.10,
            )


        lbl_jc = r"$J_c$" if i == 0 else ""
        ax.axvline(JC, color="red", linestyle="--", linewidth=1.3, label=lbl_jc)

        ax.set_title(f"$N = {int(N)}$")
        ax.set_xlabel(r"Coupling $J$")
        if i % ncols == 0:
            ax.set_ylabel("Objective (bits/s)")
        ax.margins(y=0.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


    if len(sizes) < nrows * ncols:
        for j in range(len(sizes), nrows * ncols):
            axes_flat[j].set_visible(False)


    h, l = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        h, l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout(h_pad=2.5)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_uncertainty_objective_curves_beta0.0_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
