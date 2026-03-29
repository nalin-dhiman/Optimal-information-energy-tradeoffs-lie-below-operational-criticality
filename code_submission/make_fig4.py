#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(
    
)
TBL_DIR = BASE 
RES_DIR = BASE
FIG_DIR = BASE

OPT_DIRS = [
    RES_DIR / "opt_N2000_plateau_refine_20260318_115014",
    RES_DIR / "opt_N5000_plateau_precision_20260321_192439",
    RES_DIR / "opt_N8000_scale_20260319_150812",
    RES_DIR / "opt_N10000_scale_20260319_150818",
    RES_DIR / "opt_N12000_scale_20260319_150824",
]


JC_CHI = 0.59   
JC_TAU = 0.60   


def make_fig4():

    frames = []
    for d in OPT_DIRS:
        csv = d / "opt_summary.csv"
        if csv.exists():
            frames.append(pd.read_csv(csv))
    opt = pd.concat(frames, ignore_index=True)


    pw = pd.read_csv(TBL_DIR / "plateau_width_classification.csv")


    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    

    for N in sorted(opt["N"].unique()):
        sub = opt[opt["N"] == N].sort_values("beta_E")
        ax_l.plot(
            sub["beta_E"], sub["J_star"],
            marker="o", markersize=6, linewidth=1.5,
            label=f"N={int(N)}",
        )


    ax_l.axhline(JC_CHI, color="red", linestyle="--", linewidth=1.5,
                 label=r"$J_c^{\chi} \approx 0.59$")
    ax_l.axhline(JC_TAU, color="orange", linestyle=":", linewidth=1.5,
                 label=r"$J_c^{\tau} \approx 0.60$")

    ax_l.set_xlabel(r"Energy penalty $\beta_E$")
    ax_l.set_ylabel(r"Optimal coupling $J^*$")
    ax_l.set_title(r"$\beta_E$ sensitivity")
    ax_l.set_ylim(0.42, 0.64)
    ax_l.spines["top"].set_visible(False)
    ax_l.spines["right"].set_visible(False)

    
    style = {
        0.0: dict(color="black", marker="o", label=r"$\beta_E = 0$"),
        0.2: dict(color="red", marker="s", label=r"$\beta_E = 0.2$"),
    }

    for beta, sty in style.items():
        sub = pw[pw["beta_E"] == beta].sort_values("N")
        ax_r.plot(
            sub["N"], sub["reported_width"],
            color=sty["color"], marker=sty["marker"], markersize=6,
            linewidth=1.5, label=sty["label"],
        )

    ax_r.set_xlabel(r"System size $N$")
    ax_r.set_ylabel("Plateau width (95% of max)")
    ax_r.set_title("Plateau width vs system size")
    ax_r.set_ylim(0, pw["reported_width"].max() + 0.02)
    ax_r.spines["top"].set_visible(False)
    ax_r.spines["right"].set_visible(False)


    h_l, l_l = ax_l.get_legend_handles_labels()
    h_r, l_r = ax_r.get_legend_handles_labels()

    all_h, all_l = h_l + h_r, l_l + l_r
    fig.legend(
        all_h, all_l,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=4,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    out_pdf = FIG_DIR / "Fig4_robustness_clean.pdf"
    out_png = FIG_DIR / "Fig4_robustness_clean.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    make_fig4()
