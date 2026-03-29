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

MECH_CSV = TBL_DIR / "mechanism_summary_fixed.csv"
STAB_CSV = TBL_DIR / "tuned_branch_stability_clean.csv"


def make_fig3():
    mech = pd.read_csv(MECH_CSV)
    stab = pd.read_csv(STAB_CSV)

    sizes = [2000, 5000]


    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for col, N in enumerate(sizes):

        ax_top = axes[0, col]
        m_sub = mech[mech["N"] == N]
        m_agg = m_sub.groupby("J").agg(
            L_mean=("L_recomputed", "mean"),
            L_std=("L_recomputed", "std"),
            var_mean=("var_err", "mean"),
            var_std=("var_err", "std"),
            rate_mean=("mean_rate", "mean"),
            rate_std=("mean_rate", "std"),
        ).reset_index()
        Jc = m_sub["Jc"].iloc[0]


        color_L = "black"
        ax_top.errorbar(
            m_agg["J"], m_agg["L_mean"], yerr=m_agg["L_std"],
            color=color_L, marker="o", markersize=5, linewidth=1.8,
            capsize=3, capthick=1, label=r"$\langle \mathcal{L} \rangle$",
        )
        ax_top.set_ylabel(r"Mean objective $\langle \mathcal{L} \rangle$", color=color_L)
        ax_top.tick_params(axis="y", labelcolor=color_L)


        ax_r = ax_top.twinx()
        color_var = "tab:blue"
        color_rate = "tab:green"

        ax_r.errorbar(
            m_agg["J"], m_agg["var_mean"], yerr=m_agg["var_std"],
            color=color_var, marker="^", markersize=5, linewidth=1.5,
            capsize=3, capthick=1, linestyle="--",
            label="Variance contribution",
        )
        ax_r.errorbar(
            m_agg["J"], m_agg["rate_mean"], yerr=m_agg["rate_std"],
            color=color_rate, marker="D", markersize=5, linewidth=1.5,
            capsize=3, capthick=1, linestyle="-.",
            label="Mean firing rate",
        )
        ax_r.set_ylabel("Variance / mean rate", color="dimgray")
        ax_r.tick_params(axis="y", labelcolor="dimgray")

        ax_top.set_title(f"N = {N}")
        ax_top.set_xlabel(r"Coupling $J$")
        ax_top.spines["top"].set_visible(False)
        ax_r.spines["top"].set_visible(False)


        ax_bot = axes[1, col]

        s_sub = stab[(stab["N"] == N) & (stab["beta_E"] == 0.0)]
        s_agg = s_sub.groupby("J").agg(
            eig_mean=("leading_real_eig", "mean"),
            eig_std=("leading_real_eig", "std"),
        ).reset_index()

        ax_bot.errorbar(
            s_agg["J"], s_agg["eig_mean"], yerr=s_agg["eig_std"],
            color="black", marker="o", markersize=5, linewidth=1.8,
            capsize=3, capthick=1,
            label=r"Re($\lambda_{\max}$)",
        )


        ax_bot.axhline(
            0, color="red", linestyle="--", linewidth=1.8,
            label=r"Stability boundary (Re($\lambda$)=0)",
        )

        ax_bot.set_ylabel(r"Leading real eigenvalue Re($\lambda$)")
        ax_bot.set_xlabel(r"Coupling $J$")
        ax_bot.set_title(f"N = {N}")
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)


        ymin = s_agg["eig_mean"].min() - s_agg["eig_std"].max() - 5
        ax_bot.set_ylim(ymin, 15)

 
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[0, 0].get_shared_x_axes()  
    ax_r_first = axes[0, 0].get_shared_x_axes()
   
    top_handles, top_labels = [], []
    for a in fig.axes:
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in top_labels:
                top_handles.append(h)
                top_labels.append(l)

    fig.legend(
        top_handles, top_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=5,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout(h_pad=3.0)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    out_pdf = FIG_DIR / "Fig3_mechanism_stability_clean.pdf"
    out_png = FIG_DIR / "Fig3_mechanism_stability_clean.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    make_fig3()
