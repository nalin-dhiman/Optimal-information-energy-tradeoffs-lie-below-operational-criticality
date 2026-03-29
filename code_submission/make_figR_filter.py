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


OPT_DIRS = [
    RES_DIR / "opt_N5000_plateau_precision_20260321_192439",
]

N_TARGET = 5000
BETA_TARGET = 0.0
JC = 0.60
RATE_CAP = 20.0


def make_figure():
    frames = []
    for d in OPT_DIRS:
        csv = d / "opt_rows.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    if not frames:
        print("No data found"); return

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[(raw["N"] == N_TARGET) & (raw["beta_E"] == BETA_TARGET)]


    df_A = raw[(raw["stable_flag"] == 1) & np.isfinite(raw["objective"])].copy()

    df_B = df_A[df_A["mean_rate"] < RATE_CAP].copy()

    agg_A = df_A.groupby("J")["objective"].mean().reset_index()
    agg_B = df_B.groupby("J")["objective"].mean().reset_index()

    j_star_A = agg_A.loc[agg_A["objective"].idxmax(), "J"]
    j_star_B = agg_B.loc[agg_B["objective"].idxmax(), "J"] if len(agg_B) > 0 else np.nan


    j_star = j_star_B if pd.notna(j_star_B) else j_star_A


    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4.5))


    ax.plot(agg_A["J"], agg_A["objective"],
            color="darkorange", linestyle="--", marker="s", markersize=5,
            linewidth=1.8, label="Unfiltered (stable only)")


    ax.plot(agg_B["J"], agg_B["objective"],
            color="black", linestyle="-", marker="o", markersize=5,
            linewidth=1.8, label="Filtered (rate < 20 Hz)")


    ax.axvline(j_star, color="blue", linestyle="-", linewidth=1.5,
               label=rf"$J^* = {j_star:.2f}$")


    ax.axvline(JC, color="red", linestyle="--", linewidth=1.5,
               label=rf"$J_c \approx {JC:.2f}$")

    ax.set_xlabel(r"Coupling $J$")
    ax.set_ylabel("Mean objective")
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

    stem = "FigR_filter_effect_N5000_beta0.0_clean"
    plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {stem}")


if __name__ == "__main__":
    make_figure()
