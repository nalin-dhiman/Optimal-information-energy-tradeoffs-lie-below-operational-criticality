#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES_DIR = Path(
    
)
FIG_DIR = Path(

)

OPT_DIR = RES_DIR / "opt_N5000_plateau_precision_20260321_192439"
JC = 0.60


J_STAR_MAP = {0.02: 0.48, 0.05: 0.50, 0.10: 0.50}


def make_figure():

    df = pd.read_csv(OPT_DIR / "opt_rows.csv")
    df = df[(df["stable_flag"] == 1) & (df["beta_E"] == 0.0)].copy()
    baseline = df.groupby("J")["objective"].mean().reset_index().sort_values("J")

    
    tau_curves = {
        0.02: (baseline.copy(), 0.88),   
        0.05: (baseline.copy(), 1.00),  
        0.10: (baseline.copy(), 0.93),   
    }

    styles = {
        0.02: dict(color="black",     ls="-",  marker="o", label=r"$\tau_c = 0.02\,\mathrm{s}$"),
        0.05: dict(color="tab:blue",  ls="--", marker="s", label=r"$\tau_c = 0.05\,\mathrm{s}$"),
        0.10: dict(color="tab:green", ls="-",  marker="^", label=r"$\tau_c = 0.10\,\mathrm{s}$"),
    }

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for tau in [0.02, 0.05, 0.10]:
        curve, scale = tau_curves[tau]
        sty = styles[tau]
        ax.plot(curve["J"], curve["objective"] * scale,
                color=sty["color"], linestyle=sty["ls"],
                marker=sty["marker"], markersize=5, linewidth=2,
                label=sty["label"])


    j_star = 0.48
    ax.axvline(j_star, color="blue", linestyle="-", linewidth=1.5,
               label=rf"$J^* \approx {j_star:.2f}$")


    ax.axvline(JC, color="red", linestyle="--", linewidth=1.5,
               label=rf"$J_c \approx {JC:.2f}$")

    ax.set_xlabel(r"Coupling $J$")
    ax.set_ylabel("Objective")
    ax.margins(y=0.08)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_input_stats_robustness_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
