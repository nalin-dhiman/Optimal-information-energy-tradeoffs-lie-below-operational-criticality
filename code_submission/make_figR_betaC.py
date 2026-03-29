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
BETA_C_VALS = [0.0, 0.005, 0.01]


def make_figure():
    df = pd.read_csv(OPT_DIR / "opt_rows.csv")
    df = df[(df["stable_flag"] == 1) & (df["beta_E"] == 0.0)].copy()


    df["l1_norm"] = df["theta0"].abs() + df["thetaA"].abs() + df["thetaV"].abs()


    styles = {
        0.0:   dict(color="black",    ls="-",  marker="o", label=r"$\beta_C = 0$"),
        0.005: dict(color="tab:blue",  ls="--", marker="s", label=r"$\beta_C = 0.005$"),
        0.01:  dict(color="tab:green", ls="-",  marker="^", label=r"$\beta_C = 0.01$"),
    }

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))

    j_stars = []
    for bc in BETA_C_VALS:
        col = f"obj_{bc}"
        df[col] = df["I_dec"] - bc * df["l1_norm"]
        agg = df.groupby("J")[col].mean().reset_index().sort_values("J")

        sty = styles[bc]
        ax.plot(agg["J"], agg[col],
                color=sty["color"], linestyle=sty["ls"],
                marker=sty["marker"], markersize=5, linewidth=2,
                label=sty["label"])

        j_star = agg.loc[agg[col].idxmax(), "J"]
        j_stars.append(j_star)


    j_star = j_stars[0]
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
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    stem = "FigR_betaC_sensitivity_N5000_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
