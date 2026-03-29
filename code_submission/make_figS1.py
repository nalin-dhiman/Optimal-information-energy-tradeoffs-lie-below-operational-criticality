#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

JC_DIR = Path(
    
)
FIG_DIR = Path(
   
)

CURVES_CSV = JC_DIR / "jc_curves_agg.csv"
SCALING_CSV = JC_DIR / "jc_scaling_summary.csv"


def make_figS1():
    curves = pd.read_csv(CURVES_CSV)
    scaling = pd.read_csv(SCALING_CSV)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5))

    
    N_rep = 5000
    c = curves[curves["N"] == N_rep].copy().sort_values("J")


    chi_max = c["chi_mean"].max()
    tau_max = c["tau_int_mean"].max()

    chi_norm = c["chi_mean"] / chi_max
    tau_norm = c["tau_int_mean"] / tau_max


    ax_a.plot(c["J"], chi_norm, "k-", linewidth=1.5, label=r"Normalized $\chi$")
    ax_a.plot(c["J"], tau_norm, color="gray", linestyle="--", linewidth=1.5,
              label=r"Normalized $\tau_{\mathrm{int}}$")


    jc_chi = scaling.loc[scaling["N"] == N_rep, "Jc_chi_mean"].values[0]
    jc_tau = scaling.loc[scaling["N"] == N_rep, "Jc_tau_q_mean"].values[0]

    ax_a.axvline(jc_chi, color="black", linestyle=":", linewidth=1.3,
                 label=r"$J_c^{\chi}$")
    ax_a.axvline(jc_tau, color="gray", linestyle="--", linewidth=1.3,
                 label=r"$J_c^{\tau}$")

    ax_a.set_xlabel(r"Coupling $J$")
    ax_a.set_ylabel("Normalized response")
    ax_a.set_xlim(0.40, 0.80)
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.text(0.02, 0.95, "(a)", transform=ax_a.transAxes, fontsize=11,
              va="top", fontweight="bold")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    
    s = scaling.sort_values("N")

    ax_b.plot(s["N"], s["Jc_chi_mean"], "ko-", markersize=6, linewidth=1.5,
              label=r"$J_c^{\chi}$")
    ax_b.plot(s["N"], s["Jc_tau_q_mean"], marker="s", color="gray",
              markersize=6, linewidth=1.5, linestyle="--",
              label=r"$J_c^{\tau}$")


    if "Jc_chi_boot_std" in s.columns:
        ax_b.errorbar(s["N"], s["Jc_chi_mean"],
                      yerr=s["Jc_chi_boot_std"],
                      fmt="none", ecolor="black", capsize=3, capthick=1)
    if "Jc_tau_q_boot_std" in s.columns:
        ax_b.errorbar(s["N"], s["Jc_tau_q_mean"],
                      yerr=s["Jc_tau_q_boot_std"],
                      fmt="none", ecolor="gray", capsize=3, capthick=1)

    ax_b.set_xlabel(r"System size $N$")
    ax_b.set_ylabel("Operational critical coupling")
    ax_b.set_ylim(0.57, 0.62)
    ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes, fontsize=11,
              va="top", fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)


    h_a, l_a = ax_a.get_legend_handles_labels()
    h_b, l_b = ax_b.get_legend_handles_labels()

    seen = set()
    handles, labels = [], []
    for h, l in list(zip(h_a, l_a)) + list(zip(h_b, l_b)):
        if l not in seen:
            handles.append(h)
            labels.append(l)
            seen.add(l)

    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    out_pdf = FIG_DIR / "FigS1_critical_markers_clean.pdf"
    out_png = FIG_DIR / "FigS1_critical_markers_clean.png"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    make_figS1()
