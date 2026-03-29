#!/usr/bin/env python


import argparse
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _latest_dir(glob_patterns):

    candidates = []
    for pat in glob_patterns:
        candidates.extend(Path(".").glob(pat))
    candidates = [p for p in candidates if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def plot_jc_scaling(df: pd.DataFrame, out_path: Path):

    plt.figure(figsize=(6, 5))


    if "Jc_chi_mean" in df.columns:
        plt.errorbar(
            df["N"],
            df["Jc_chi_mean"],
            yerr=df["Jc_chi_std"] if "Jc_chi_std" in df.columns else None,
            fmt="o-",
            label=r"$J_c^\chi$",
        )

    if "Jc_tau_q_mean" in df.columns:
        plt.errorbar(
            df["N"],
            df["Jc_tau_q_mean"],
            yerr=df["Jc_tau_std"] if "Jc_tau_std" in df.columns else None,
            fmt="s-",
            label=r"$J_c^\tau$ (q-peak)",
        )
    elif "Jc_tau_mean" in df.columns:
        plt.errorbar(
            df["N"],
            df["Jc_tau_mean"],
            yerr=df["Jc_tau_std"] if "Jc_tau_std" in df.columns else None,
            fmt="s-",
            label=r"$J_c^\tau$",
        )


    if "Jc_chi" in df.columns:
        plt.plot(df["N"], df["Jc_chi"], "o-", label=r"$J_c^\chi$ (legacy)")
    if "Jc_tau" in df.columns:
        plt.plot(df["N"], df["Jc_tau"], "s-", label=r"$J_c^\tau$ (legacy)")

    plt.xscale("log")
    plt.xlabel(r"Network Size $N$")
    plt.ylabel(r"Critical Coupling $J_c^{\mathrm{ren}}$")
    plt.title("Renormalized Criticality Scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_opt_summary(df: pd.DataFrame, out_dir: Path):
    
    if len(df) == 0:
        return

    for (N, tau_c), sub in df.groupby(["N", "tau_c"]):
        sub = sub.sort_values("beta_E")


        if "J_star" in sub.columns:
            plt.figure(figsize=(7, 5))
            plt.plot(sub["beta_E"], sub["J_star"], marker="o")
            if "Jc_used" in sub.columns and sub["Jc_used"].notna().any():
                plt.axhline(float(sub["Jc_used"].dropna().iloc[0]), linestyle="--")
            plt.xlabel(r"Energy Penalty $\beta_E$")
            plt.ylabel(r"Optimal Coupling $J^*$")
            plt.title(f"Optimal Coupling vs Energy Penalty (N={N}, tau_c={tau_c})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"opt_Jstar_vs_betaE_N{N}_tau{tau_c}.png", dpi=300)
            plt.close()


        if "Delta_star" in sub.columns and sub["Delta_star"].notna().any():
            plt.figure(figsize=(7, 5))
            plt.plot(sub["beta_E"], sub["Delta_star"], marker="o")
            plt.axhline(0.0, linestyle="--")
            plt.xlabel(r"Energy Penalty $\beta_E$")
            plt.ylabel(r"Optimal Distance $\Delta^* = J_c - J^*$")
            plt.title(f"Distance to Criticality vs Energy Penalty (N={N}, tau_c={tau_c})")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"opt_Deltastar_vs_betaE_N{N}_tau{tau_c}.png", dpi=300)
            plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate summary figures for the PRL pipeline.")
    parser.add_argument("--jc_outdir", type=str, default=None, help="Directory containing jc_scaling_summary.csv")
    parser.add_argument("--opt_outdir", type=str, default=None, help="Directory containing opt_summary.csv (or opt_rows.csv)")
    parser.add_argument("--outdir", type=str, default="results/figs", help="Output directory for figures")
    parser.add_argument("--auto", action="store_true", help="Auto-detect latest jc_* and opt_* results dirs")
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jc_dir = Path(args.jc_outdir) if args.jc_outdir else None
    opt_dir = Path(args.opt_outdir) if args.opt_outdir else None

    if args.auto:
        if jc_dir is None:
            jc_dir = _latest_dir(["results/jc_refine_*", "results/jc_coarse_*", "results/jc_*"])
        if opt_dir is None:
            opt_dir = _latest_dir(["results/opt_prl_*", "results/opt_*"])
        print(f"[auto] jc_dir={jc_dir}")
        print(f"[auto] opt_dir={opt_dir}")

    # --- Jc scaling figure ---
    if jc_dir is not None:
        jc_csv = jc_dir / "jc_scaling_summary.csv"
        if jc_csv.exists():
            df_jc = pd.read_csv(jc_csv)
            plot_jc_scaling(df_jc, out_dir / "jc_scaling_summary.png")
            print(f"Wrote {out_dir / 'jc_scaling_summary.png'}")
        else:
            print(f"WARNING: {jc_csv} not found; skipping Jc plot.")

    # --- Optimization summary figures ---
    if opt_dir is not None:
        opt_sum = opt_dir / "opt_summary.csv"
        if opt_sum.exists():
            df_opt = pd.read_csv(opt_sum)
            plot_opt_summary(df_opt, out_dir)
            print(f"Wrote opt summary figures into {out_dir}")
        else:
            print(f"WARNING: {opt_sum} not found; skipping opt plots.")

    print("Done.")


if __name__ == "__main__":
    main()
