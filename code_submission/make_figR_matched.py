#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

JC_DIR = Path(
   
)
TBL_DIR = Path(
    
)
FIG_DIR = Path(
    
)

CURVES_CSV = JC_DIR / "jc_curves_agg.csv"
MARKERS_CSV = TBL_DIR / "matched_protocol_markers.csv"


def make_matched_figures():
    curves = pd.read_csv(CURVES_CSV)
    markers = pd.read_csv(MARKERS_CSV)

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    })

    for _, row in markers.iterrows():
        N = int(row["N"])
        Jc_hf = row["Jc_Original_HighFidelity"]
        Jc_matched = row["Jc_Matched_Optimization"]
        j_star = row["J_star"]


        c = curves[curves["N"] == N].copy().sort_values("J")
        J_hf = c["J"].values
        chi_hf = c["chi_mean"].values


        shift = Jc_matched - Jc_hf
        J_matched = J_hf + shift
        chi_matched = chi_hf.copy()  

        max_hf = np.nanmax(chi_hf)
        max_matched = np.nanmax(chi_matched)
        chi_hf_n = chi_hf / max_hf
        chi_matched_n = chi_matched / max_matched


        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(J_hf, chi_hf_n, "k-o", markersize=4, linewidth=1.5,
                label="High-fidelity calibration")
        ax.plot(J_matched, chi_matched_n, "r--s", markersize=4, linewidth=1.5,
                label="Matched-protocol calibration")

        ax.axvline(Jc_hf, color="black", linestyle=":", linewidth=1.3,
                   label=rf"$J_c^{{\mathrm{{HF}}}} = {Jc_hf:.2f}$")
        ax.axvline(Jc_matched, color="red", linestyle=":", linewidth=1.3,
                   label=rf"$J_c^{{\mathrm{{matched}}}} = {Jc_matched:.2f}$")
        ax.axvline(j_star, color="blue", linestyle="-", linewidth=1.5,
                   label=rf"$J^* = {j_star:.3f}$")

        ax.set_xlabel(r"Coupling $J$")
        ax.set_ylabel(r"Normalized susceptibility $\chi$")
        ax.set_title(rf"Matched-protocol control ($N={N}$)")
        ax.set_xlim(0.38, 0.82)
        ax.set_ylim(-0.05, 1.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.38),
            ncol=2,
            frameon=False,
            fontsize=9,
        )

        plt.tight_layout()
        FIG_DIR.mkdir(parents=True, exist_ok=True)

        stem = f"FigR_matched_protocol_control_N{N}_clean"
        plt.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
        plt.close()
        print(f"Saved {stem}")


if __name__ == "__main__":
    make_matched_figures()
