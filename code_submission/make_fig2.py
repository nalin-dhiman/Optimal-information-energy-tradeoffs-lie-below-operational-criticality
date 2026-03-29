#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(
    
)
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = Path(
   
)


UNCERT_CSV = Path(
    
)


def make_fig2():
    df = pd.read_csv(UNCERT_CSV)

    JC = 0.60  

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.5))


    style = {
        0.0: dict(color="black", marker="o", label=r"$\beta_E = 0$"),
        0.2: dict(color="red", marker="s", label=r"$\beta_E = 0.2$"),
    }

    for beta, sty in style.items():
        sub = df[df["beta_E"] == beta].sort_values("N")
        N = sub["N"].values


        j_mean = sub["J_star_mean"].values
        j_err = sub["J_star_std"].values
        ax_l.errorbar(
            N, j_mean, yerr=j_err,
            color=sty["color"], marker=sty["marker"], markersize=7,
            linewidth=1.8, capsize=3, capthick=1.2,
            label=sty["label"],
        )


        d_mean = sub["Delta_star_mean"].values
        d_err = sub["Delta_star_std"].values
        ax_r.errorbar(
            N, d_mean, yerr=d_err,
            color=sty["color"], marker=sty["marker"], markersize=7,
            linewidth=1.8, capsize=3, capthick=1.2,
            label=sty["label"],
        )


    ax_l.axhline(JC, color="red", linestyle="--", linewidth=1.5, label=r"$J_c$")
    ax_l.annotate(
        r"$J_c \approx 0.60$",
        xy=(df["N"].max(), JC),
        xytext=(10, 6), textcoords="offset points",
        fontsize=10, color="red", va="bottom",
    )
    ax_l.set_xlabel(r"System size $N$")
    ax_l.set_ylabel(r"Optimal coupling $J^*$")

    j_all = df["J_star_mean"].values
    y_lo = min(j_all) - 0.03
    y_hi = JC + 0.04
    ax_l.set_ylim(y_lo, y_hi)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)


    ax_r.axhline(0, color="gray", linestyle="--", linewidth=1.2)
    ax_r.set_xlabel(r"System size $N$")
    ax_r.set_ylabel(r"Normalized offset $\Delta^* = (J_c - J^*)/J_c$")

    d_all = df["Delta_star_mean"].values
    d_lo = -0.02
    d_hi = max(d_all) + 0.04
    ax_r.set_ylim(d_lo, d_hi)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)


    handles_l, labels_l = ax_l.get_legend_handles_labels()
    fig.legend(
        handles_l, labels_l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=False,
        fontsize=11,
    )

    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = FIG_DIR / "Fig2_regime_summary_clean.pdf"
    out_png = FIG_DIR / "Fig2_regime_summary_clean.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    make_fig2()
