#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def select_peak_jc(J_vals: np.ndarray, metric_vals: np.ndarray, stable_mask: np.ndarray) -> tuple[float, bool]:
    
    valid_indices = np.where(stable_mask)[0]
    if len(valid_indices) == 0:
        return np.nan, True

    last_stable_idx = valid_indices[-1]

    metric_valid = np.full_like(metric_vals, -np.inf, dtype=float)
    metric_valid[valid_indices] = metric_vals[valid_indices]

    peak_idx = int(np.argmax(metric_valid))
    if np.isinf(metric_valid[peak_idx]):
        return np.nan, True

    Jc = float(J_vals[peak_idx])


    edge_peak = peak_idx >= (last_stable_idx - 1)
    return Jc, edge_peak


def select_tau_quantile_jc(J_vals: np.ndarray, tau_vals: np.ndarray, stable_mask: np.ndarray, q: float) -> tuple[float, bool, float]:
   
    valid_indices = np.where(stable_mask)[0]
    if len(valid_indices) == 0:
        return np.nan, True, 0.0

    J_stable = J_vals[valid_indices]
    tau_stable = tau_vals[valid_indices].copy()


    tau_stable = np.nan_to_num(tau_stable, nan=0.0, posinf=0.0, neginf=0.0)

    tau_max = float(np.max(tau_stable))
    if tau_max <= 0.0:
        return np.nan, True, tau_max

    target = q * tau_max


    crossing_local = int(np.argmax(tau_stable >= target))
    Jc = float(J_stable[crossing_local])


    edge_warning = crossing_local >= (len(valid_indices) - 2)
    return Jc, edge_warning, tau_max


def plot_diagnostic(agg: pd.DataFrame, stable_mask: np.ndarray,
                    Jc_chi: float, Jc_tau_q: float, N: int, out_path: Path, tau_q: float):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    J = agg["J"].to_numpy()
    chi_m = agg["chi_mean"].to_numpy()
    chi_s = agg["chi_std"].to_numpy()
    tau_m = agg["tau_int_mean"].to_numpy()
    tau_s = agg["tau_int_std"].to_numpy()


    ax1.errorbar(J[stable_mask], chi_m[stable_mask], yerr=chi_s[stable_mask], fmt="o-", label="Stable")
    ax1.errorbar(J[~stable_mask], chi_m[~stable_mask], yerr=chi_s[~stable_mask], fmt="x", color="red", label="Unstable")
    if not np.isnan(Jc_chi):
        ax1.axvline(Jc_chi, color="k", linestyle="--", label=f"Jc_chi = {Jc_chi:.3f}")
    ax1.set_xlabel("J")
    ax1.set_ylabel(r"$\chi$")
    ax1.set_title(f"Susceptibility N={N}")
    ax1.legend()


    ax2.errorbar(J[stable_mask], tau_m[stable_mask], yerr=tau_s[stable_mask], fmt="s-", color="orange", label="Stable")
    ax2.errorbar(J[~stable_mask], tau_m[~stable_mask], yerr=tau_s[~stable_mask], fmt="x", color="red", label="Unstable")
    if not np.isnan(Jc_tau_q):
        ax2.axvline(Jc_tau_q, color="k", linestyle="--", label=f"Jc_tau(q={tau_q:.2f}) = {Jc_tau_q:.3f}")
    ax2.set_xlabel("J")
    ax2.set_ylabel(r"$\tau_{\mathrm{int}}$ (s)")
    ax2.set_title(f"Correlation Time N={N}")
    ax2.legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_jc_scaling(df_summary: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(6, 5))
    if "Jc_chi_mean" in df_summary.columns:
        plt.errorbar(df_summary["N"], df_summary["Jc_chi_mean"],
                     yerr=df_summary.get("Jc_chi_boot_std", df_summary.get("Jc_chi_std", None)),
                     fmt="o-", label=r"$J_c^\chi$ (Susceptibility)")
    if "Jc_tau_q_mean" in df_summary.columns:
        plt.errorbar(df_summary["N"], df_summary["Jc_tau_q_mean"],
                     yerr=df_summary.get("Jc_tau_q_boot_std", df_summary.get("Jc_tau_std", None)),
                     fmt="s-", label=r"$J_c^\tau$ (Quantile)")

    plt.xscale("log")
    plt.xlabel(r"Network Size $N$")
    plt.ylabel(r"Critical Coupling $J_c^{\mathrm{ren}}$")
    plt.title("Renormalized Criticality Scaling")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def bootstrap_jc(df_N: pd.DataFrame, stable_frac_threshold: float, tau_q: float,
                 n_boot: int, rng: np.random.Generator) -> tuple[float, float, float, float]:

    seeds = np.array(sorted(df_N["seed"].unique()))
    if len(seeds) == 0:
        return np.nan, np.nan, np.nan, np.nan

    jc_chi_samples = []
    jc_tau_samples = []

    for _ in range(n_boot):
        samp = rng.choice(seeds, size=len(seeds), replace=True)
        df_b = pd.concat([df_N[df_N["seed"] == s] for s in samp], ignore_index=True)

        agg = df_b.groupby("J").agg(
            chi_mean=("chi", "mean"),
            chi_std=("chi", "std"),
            tau_int_mean=("tau_int", "mean"),
            tau_int_std=("tau_int", "std"),
            mean_rate_mean=("mean_rate", "mean"),
            mean_rate_std=("mean_rate", "std"),
            stable_frac=("stable_flag", "mean"),
        ).reset_index()

        stable_mask = (agg["stable_frac"].to_numpy() >= stable_frac_threshold)

        J = agg["J"].to_numpy()
        chi = agg["chi_mean"].to_numpy()
        tau = agg["tau_int_mean"].to_numpy()

        jc_chi, _ = select_peak_jc(J, chi, stable_mask)
        jc_tau, _, _ = select_tau_quantile_jc(J, tau, stable_mask, tau_q)

        if not np.isnan(jc_chi):
            jc_chi_samples.append(jc_chi)
        if not np.isnan(jc_tau):
            jc_tau_samples.append(jc_tau)

    def _mean_std(x):
        if len(x) == 0:
            return np.nan, np.nan
        x = np.array(x, dtype=float)
        return float(np.mean(x)), float(np.std(x))  
    chi_m, chi_s = _mean_std(jc_chi_samples)
    tau_m, tau_s = _mean_std(jc_tau_samples)
    return chi_m, chi_s, tau_m, tau_s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, type=str, help="Directory containing rows_*.csv")
    parser.add_argument("--pattern", default="rows_*.csv", type=str)

    parser.add_argument("--tau_quantile", type=float, default=0.9,
                        help="Quantile q for tau-based Jc: first J where tau >= q * max(tau) (stable points).")
    parser.add_argument("--stable_frac_threshold", type=float, default=(2/3),
                        help="Stable threshold for aggregated curves. Coarse (3 seeds): 2/3 recommended; Refine (5 seeds): 0.6 recommended.")
    parser.add_argument("--allow_partial", action="store_true",
                        help="Allow merging even if partitions appear incomplete.")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap Jc uncertainties across seeds.")
    parser.add_argument("--n_boot", type=int, default=200, help="Number of bootstrap resamples.")
    parser.add_argument("--boot_seed", type=int, default=123, help="RNG seed for bootstrap.")

    args = parser.parse_args()

    out_dir = Path(args.outdir)
    csv_files = list(out_dir.glob(args.pattern))

    if not csv_files:
        print(f"Error: No files matching '{args.pattern}' found in {out_dir}")
        return

    print(f"Found {len(csv_files)} partition files. Merging...", flush=True)

    df_list = [pd.read_csv(f) for f in csv_files]
    df_combined = pd.concat(df_list).drop_duplicates(subset=["N", "J", "seed", "pass_id"]).copy()


    pass_weights = {"coarse": 1, "refine": 2}
    df_combined["pass_w"] = df_combined["pass_id"].map(pass_weights).fillna(1)
    df_combined = df_combined.sort_values("pass_w").drop_duplicates(subset=["N", "J", "seed"], keep="last")
    df_combined = df_combined.sort_values(["N", "J", "seed"])

    N_list = sorted(df_combined["N"].unique())
    seeds_all = sorted(df_combined["seed"].unique())


    expected = len(N_list) * len(seeds_all)
    found = len(csv_files)
    if (found < expected) and (not args.allow_partial):
        print(f"ERROR: Found {found}/{expected} partitions (N={len(N_list)} seeds={len(seeds_all)}). "
              f"Aborting merge. Use --allow_partial to override.", flush=True)
        return

    summary_rows = []
    agg_all_rows = []

    rng = np.random.default_rng(args.boot_seed)

    for N in N_list:
        df_N = df_combined[df_combined["N"] == N].copy()


        agg = df_N.groupby("J").agg(
            chi_mean=("chi", "mean"),
            chi_std=("chi", "std"),
            tau_int_mean=("tau_int", "mean"),
            tau_int_std=("tau_int", "std"),
            mean_rate_mean=("mean_rate", "mean"),
            mean_rate_std=("mean_rate", "std"),
            stable_frac=("stable_flag", "mean"),
        ).reset_index()

        agg["N"] = N
        agg_all_rows.append(agg)

        stable_mask = (agg["stable_frac"].to_numpy() >= args.stable_frac_threshold)


        J = agg["J"].to_numpy()
        chi = agg["chi_mean"].to_numpy()
        tau = agg["tau_int_mean"].to_numpy()

        Jc_chi_agg, edge_peak_chi = select_peak_jc(J, chi, stable_mask)


        Jc_tau_q_agg, edge_warn_tau, tau_max = select_tau_quantile_jc(J, tau, stable_mask, args.tau_quantile)


        tau_nan_frac = float(np.mean(pd.isna(agg["tau_int_mean"].to_numpy())))


        Jc_chi_boot_std = np.nan
        Jc_tau_boot_std = np.nan
        if args.bootstrap:
            chi_m, chi_s, tau_m, tau_s = bootstrap_jc(df_N, args.stable_frac_threshold, args.tau_quantile, args.n_boot, rng)

            Jc_chi_boot_std = chi_s
            Jc_tau_boot_std = tau_s


        plot_path = out_dir / f"curves_N{N}.png"
        plot_diagnostic(agg, stable_mask, Jc_chi_agg, Jc_tau_q_agg, int(N), plot_path, args.tau_quantile)

        summary_rows.append({
            "N": int(N),
            "Jc_chi_mean": Jc_chi_agg,
            "Jc_chi_std": 0.0,  
            "Jc_chi_boot_std": Jc_chi_boot_std,
            "edge_peak_chi": bool(edge_peak_chi),

            "Jc_tau_q_mean": Jc_tau_q_agg,
            "Jc_tau_std": 0.0,  
            "Jc_tau_q_boot_std": Jc_tau_boot_std,
            "edge_warn_tau": bool(edge_warn_tau),

            "tau_max": float(tau_max),
            "tau_nan_frac": tau_nan_frac,
            "stable_frac_threshold": float(args.stable_frac_threshold),
            "tau_quantile": float(args.tau_quantile),
        })

    df_agg_all = pd.concat(agg_all_rows, ignore_index=True)
    agg_path = out_dir / "jc_curves_agg.csv"
    df_agg_all.to_csv(agg_path, index=False)
    print(f"Saved aggregated curves to {agg_path}", flush=True)

    df_summary = pd.DataFrame(summary_rows)
    sum_path = out_dir / "jc_scaling_summary.csv"
    df_summary.to_csv(sum_path, index=False)
    print(f"Saved scaling summary to {sum_path}", flush=True)

    scale_plot_path = out_dir / "jc_scaling.png"
    plot_jc_scaling(df_summary, scale_plot_path)


    print("\n" + "=" * 40, flush=True)
    print("SANITY CHECK RESULTS", flush=True)
    print("=" * 40, flush=True)
    print("Jc_ren summary:", flush=True)
    for _, row in df_summary.iterrows():
        print(
            f"N={int(row['N'])}: "
            f"Jc_chi={row['Jc_chi_mean']:.3f}, "
            f"Jc_tau(q={row['tau_quantile']:.2f})={row['Jc_tau_q_mean']:.3f}, "
            f"tau_nan_frac={row['tau_nan_frac']:.2f}",
            flush=True
        )

    edge_any = df_summary["edge_peak_chi"].any() or df_summary["edge_warn_tau"].any()
    print(f"Edge warnings triggered: {edge_any}", flush=True)

    confidence = "high"
    if edge_any:
        confidence = "medium"  
    if df_summary["tau_nan_frac"].max() > 0.3:
        confidence = "medium"
    print(f"Confidence: {confidence}", flush=True)


if __name__ == "__main__":
    main()
