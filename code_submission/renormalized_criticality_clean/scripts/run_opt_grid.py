#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import load_config, safe_save_run_manifest
from src.stimulus import generate_ou_stimulus, effective_sampling_dt
from src.optimize import optimize_theta


def _to_python(x: Any) -> Any:

    try:
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    return x


def _parse_csv_list(s: str | None, cast=float) -> List:
    if s is None:
        return []
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue
        out.append(cast(tok))
    return out


def _load_jc_center(jc_outdir: str | None, N: int, source: str) -> float:
    
    if jc_outdir is None:
        return float("nan")

    jc_csv = Path(jc_outdir) / "jc_scaling_summary.csv"
    if not jc_csv.exists():
        print(f"WARNING: jc_outdir provided but file not found: {jc_csv}")
        return float("nan")

    df = pd.read_csv(jc_csv)
    if len(df) == 0 or "N" not in df.columns:
        print(f"WARNING: jc_scaling_summary.csv missing data or 'N' column: {jc_csv}")
        return float("nan")

    if N in df["N"].values:
        row = df[df["N"] == N].iloc[0]
    else:
        nearest_N = df.iloc[(df["N"] - N).abs().argsort()[:1]]["N"].values[0]
        print(f"WARNING: Exact N={N} not found in {jc_csv}. Using nearest N={nearest_N}.")
        row = df[df["N"] == nearest_N].iloc[0]

    if source == "tau":
        for key in ["Jc_tau_q_mean", "Jc_tau_mean", "Jc_tau", "Jc_chi_mean", "Jc_chi"]:
            if key in row and pd.notna(row[key]):
                return float(row[key])
    else:  # chi
        for key in ["Jc_chi_mean", "Jc_chi", "Jc_tau_q_mean", "Jc_tau_mean", "Jc_tau"]:
            if key in row and pd.notna(row[key]):
                return float(row[key])

    return float("nan")


def _build_J_grid(
    Jc_used: float,
    J_window: float,
    J_points: int,
    J_extra_offsets: List[float],
    fallback: Iterable[float],
    explicit_j_list: List[float] | None = None,
) -> List[float]:
   
    if explicit_j_list:
        return sorted({float(np.round(j, 12)) for j in explicit_j_list})

    if np.isfinite(Jc_used):
        J_base = np.linspace(max(0.0, Jc_used - J_window), Jc_used + J_window, int(J_points))
        J_extra = Jc_used + np.array(J_extra_offsets, dtype=float)
        J_concat = np.concatenate([J_base, J_extra])
        J_grid = sorted({float(np.round(j, 12)) for j in J_concat})
        return J_grid

    return [float(j) for j in fallback]


def _append_row_csv(path: Path, row: Dict[str, Any], fieldnames: List[str]) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow({k: _to_python(row.get(k, "")) for k in fieldnames})
        f.flush()


def _load_done_keys(opt_rows_csv: Path) -> set[Tuple]:
    
    if not opt_rows_csv.exists():
        return set()
    try:
        df = pd.read_csv(opt_rows_csv, usecols=["N", "seed", "tau_c", "beta_E", "beta_C", "J"])
    except Exception:
        return set()

    keys = set()
    for r in df.itertuples(index=False):
        keys.add((int(r.N), int(r.seed), float(r.tau_c), float(r.beta_E), float(r.beta_C), float(r.J)))
    return keys


def _apply_summary_filter(df: pd.DataFrame, rate_cap: float | None) -> pd.DataFrame:
    
    out = df.copy()
    out = out[out["stable_flag"] == 1]
    out = out[np.isfinite(out["objective"])]
    out = out[np.isfinite(out["mean_rate"])]

    if rate_cap is not None and np.isfinite(rate_cap):
        out = out[out["mean_rate"] < rate_cap]

    return out


def plot_prl_figures(df_rows: pd.DataFrame, df_summary: pd.DataFrame, out_dir: Path, rate_cap: float | None) -> None:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df_rows) == 0:
        print("WARNING: empty df_rows, skipping plots.")
        return

    df_plot = _apply_summary_filter(df_rows, rate_cap)
    if len(df_plot) == 0:
        print("WARNING: no valid filtered rows for plotting; skipping plots.")
        return

    Ns = sorted(df_plot["N"].unique())
    tau_cs = sorted(df_plot["tau_c"].unique())

    for N in Ns:
        for tau_c in tau_cs:
            sub_df = df_plot[(df_plot["N"] == N) & (df_plot["tau_c"] == tau_c)]
            if len(sub_df) == 0:
                continue

            j_agg = (
                sub_df.groupby(["beta_E", "J"])
                .agg(
                    obj_mean=("objective", "mean"),
                    obj_std=("objective", "std"),
                    Delta_mean=("Delta", "mean"),
                    n_valid=("objective", "size"),
                )
                .reset_index()
            )


            plt.figure(figsize=(8, 6))
            for beta_E in sorted(sub_df["beta_E"].unique()):
                trace = j_agg[j_agg["beta_E"] == beta_E].sort_values("J")
                plt.plot(trace["J"], trace["obj_mean"], marker="o", label=rf"$\beta_E = {beta_E}$")

            Jc_val = sub_df["Jc_used"].dropna().iloc[0] if sub_df["Jc_used"].notna().any() else np.nan
            if np.isfinite(Jc_val):
                plt.axvline(Jc_val, color="r", linestyle="--", label=r"$J_c$ used")

            plt.xlabel(r"Coupling $J$")
            plt.ylabel(r"Mean Objective $\langle \mathcal{L} \rangle$")
            title = rf"Optimization Landscape (N={N}, $\tau_c$={tau_c})"
            if rate_cap is not None:
                title += rf", rate< {rate_cap}"
            plt.title(title)
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / f"objective_vs_J_N{N}_tau{tau_c}.png", dpi=300)
            plt.close()


            if np.isfinite(Jc_val):
                plt.figure(figsize=(8, 6))
                for beta_E in sorted(sub_df["beta_E"].unique()):
                    trace = j_agg[j_agg["beta_E"] == beta_E].sort_values("Delta_mean")
                    plt.plot(trace["Delta_mean"], trace["obj_mean"], marker="o", label=rf"$\beta_E = {beta_E}$")

                plt.axvline(0.0, color="r", linestyle="--", label=r"$\Delta = 0$ ($J_c$)")
                plt.xlabel(r"Distance from Criticality $\Delta = J_c - J$")
                plt.ylabel(r"Mean Objective $\langle \mathcal{L} \rangle$")
                title = rf"Collapse vs $\Delta$ (N={N}, $\tau_c$={tau_c})"
                if rate_cap is not None:
                    title += rf", rate< {rate_cap}"
                plt.title(title)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"objective_vs_Delta_N{N}_tau{tau_c}.png", dpi=300)
                plt.close()


            summary_sub = (
                df_summary[(df_summary["N"] == N) & (df_summary["tau_c"] == tau_c)]
                .dropna(subset=["J_star"])
                .sort_values("beta_E")
            )
            if len(summary_sub) > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(summary_sub["beta_E"], summary_sub["J_star"], marker="s")
                if np.isfinite(Jc_val):
                    plt.axhline(Jc_val, color="r", linestyle="--", label=r"$J_c$")
                    plt.legend()
                plt.xlabel(r"Energy Penalty $\beta_E$")
                plt.ylabel(r"Optimal Coupling $J^*$")
                title = rf"$J^*$ vs $\beta_E$ (N={N}, $\tau_c$={tau_c})"
                if rate_cap is not None:
                    title += rf", rate< {rate_cap}"
                plt.title(title)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_dir / f"Jstar_vs_betaE_N{N}_tau{tau_c}.png", dpi=300)
                plt.close()

                if np.isfinite(Jc_val):
                    plt.figure(figsize=(8, 6))
                    plt.plot(summary_sub["beta_E"], summary_sub["Delta_star"], marker="s")
                    plt.axhline(0.0, color="r", linestyle="--", label=r"$\Delta = 0$")
                    plt.xlabel(r"Energy Penalty $\beta_E$")
                    plt.ylabel(r"Optimal Distance $\Delta^* = J_c - J^*$")
                    title = rf"$\Delta^*$ vs $\beta_E$ (N={N}, $\tau_c$={tau_c})"
                    if rate_cap is not None:
                        title += rf", rate< {rate_cap}"
                    plt.title(title)
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(out_dir / f"Deltastar_vs_betaE_N{N}_tau{tau_c}.png", dpi=300)
                    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--outdir", default=None)


    parser.add_argument("--jc_outdir", type=str, default=None, help="Directory containing jc_scaling_summary.csv")
    parser.add_argument("--N", type=int, default=None, help="Run a single N (otherwise uses config)")
    parser.add_argument("--J_center_source", type=str, choices=["tau", "chi"], default="tau")
    parser.add_argument("--J_window", type=float, default=0.20)
    parser.add_argument("--J_points", type=int, default=13)
    parser.add_argument("--J_extra", type=str, default="0.00,-0.02,-0.01,0.01,0.02")
    parser.add_argument("--J_list", type=str, default=None, help="Explicit comma-separated J list, overrides generated grid")


    parser.add_argument("--seeds", type=str, default=None, help="Comma list override, e.g. '1,2,3'")
    parser.add_argument("--tau_c_list", type=str, default=None, help="Comma list override, e.g. '0.05,0.1'")
    parser.add_argument("--beta_E_list", type=str, default=None, help="Comma list override, e.g. '0.01,0.1'")
    parser.add_argument("--beta_C", type=float, default=None)


    parser.add_argument("--T_opt", type=float, default=None, help="Override optimization simulation horizon T")
    parser.add_argument("--dt_opt", type=float, default=None, help="Override optimization timestep dt")
    parser.add_argument("--n_restarts", type=int, default=None)
    parser.add_argument("--maxiter", type=int, default=None)


    parser.add_argument("--resume", action="store_true", help="Resume if opt_rows.csv exists (skip completed points)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing opt_rows.csv and opt_summary.csv")
    parser.add_argument("--progress_every", type=int, default=10, help="Print progress every K points")
    parser.add_argument("--rate_cap", type=float, default=None, help="Optional stricter filter for summary/plots: mean_rate < rate_cap")

    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    config = load_config(args.config)


    if args.outdir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"results/opt_{ts}"
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt_rows_csv = out_dir / "opt_rows.csv"
    opt_summary_csv = out_dir / "opt_summary.csv"

    if args.overwrite:
        for p in [opt_rows_csv, opt_summary_csv]:
            if p.exists():
                p.unlink()

   
    if args.smoke_test:
        N_list = [50]
        T = 1.0
        dt = 0.001
        J_points = 3
        seeds = [1]
        tau_c_list = [0.05]
        beta_E_list = [0.01]
        beta_C = args.beta_C if args.beta_C is not None else config.get("optimization", {}).get("beta_C", 0.1)
        n_restarts = 1
        maxiter_val = 1
        fallback_J = np.linspace(0.3, 0.7, J_points).tolist()
        explicit_j_list = None
    else:
        if args.N is not None:
            N_list = [int(args.N)]
        else:
            N_list = config.get("simulation", {}).get("N_list", [config["simulation"]["N"]])

        T = float(args.T_opt) if args.T_opt is not None else float(config.get("optimization", {}).get("T_opt", config["simulation"]["T"]))
        dt = float(args.dt_opt) if args.dt_opt is not None else float(config.get("optimization", {}).get("dt_opt", config["simulation"]["dt"]))

        if args.seeds is not None:
            seeds = [int(s) for s in _parse_csv_list(args.seeds, cast=int)]
        else:
            seeds = config.get("sweep", {}).get("seeds", config.get("simulation", {}).get("seeds", [42]))

        if args.tau_c_list is not None:
            tau_c_list = [float(x) for x in _parse_csv_list(args.tau_c_list, cast=float)]
        else:
            tau_c_list = config.get("stimulus", {}).get("tau_c_list", [float(config.get("stimulus", {}).get("tau_c", 0.05))])

        if args.beta_E_list is not None:
            beta_E_list = [float(x) for x in _parse_csv_list(args.beta_E_list, cast=float)]
        else:
            beta_E_list = config.get("optimization", {}).get("beta_E_list", [float(config.get("optimization", {}).get("beta_E", 0.01))])

        beta_C = float(args.beta_C) if args.beta_C is not None else float(config.get("optimization", {}).get("beta_C", 0.005))
        n_restarts = int(args.n_restarts) if args.n_restarts is not None else int(config.get("optimization", {}).get("n_restarts", 3))
        maxiter_val = int(args.maxiter) if args.maxiter is not None else int(config.get("optimization", {}).get("maxiter", 50))

        fallback_J = config.get("optimization", {}).get("J_list", np.linspace(0.1, 2.5, int(args.J_points)).tolist())
        explicit_j_list = [float(x) for x in _parse_csv_list(args.J_list, cast=float)] if args.J_list is not None else None

    J_extra_offsets = [float(x) for x in _parse_csv_list(args.J_extra, cast=float)]

    print("=" * 80)
    print("Running Optimization Grid")
    print(f"out_dir: {out_dir}")
    print(f"smoke_test: {args.smoke_test}")
    print(f"N_list: {N_list}")
    print(f"T={T}  dt={dt}")
    print(f"seeds: {seeds}")
    print(f"tau_c_list: {tau_c_list}")
    print(f"beta_E_list: {beta_E_list}")
    print(f"beta_C: {beta_C}")
    print(f"n_restarts: {n_restarts}  maxiter: {maxiter_val}")
    print(f"jc_outdir: {args.jc_outdir}  J_center_source: {args.J_center_source}")
    print(f"J_list override: {explicit_j_list if explicit_j_list else 'None'}")
    print(f"rate_cap for summary/plots: {args.rate_cap}")
    print("=" * 80, flush=True)

    done_keys = set()
    if args.resume:
        done_keys = _load_done_keys(opt_rows_csv)
        if done_keys:
            print(f"[resume] found {len(done_keys)} completed points in {opt_rows_csv}", flush=True)

    fieldnames = [
        "N",
        "seed",
        "tau_c",
        "beta_E",
        "beta_C",
        "J",
        "Jc_used",
        "Delta",
        "theta0",
        "thetaV",
        "thetaA",
        "I_dec",
        "mean_rate",
        "objective",
        "stable_flag",
        "error_msg",
    ]

    approx_J = len(explicit_j_list) if explicit_j_list else (int(args.J_points) + len(J_extra_offsets))
    total_pts = len(N_list) * len(seeds) * len(tau_c_list) * len(beta_E_list) * approx_J
    completed = 0

    
    for N in N_list:
        print(f"\n--- N={N} ---", flush=True)

        Jc_used = _load_jc_center(args.jc_outdir, N, args.J_center_source)
        print(f"Jc_used: {Jc_used}", flush=True)

        J_grid = _build_J_grid(
            Jc_used=Jc_used,
            J_window=float(args.J_window),
            J_points=int(args.J_points),
            J_extra_offsets=J_extra_offsets,
            fallback=fallback_J,
            explicit_j_list=explicit_j_list,
        )

        with open(out_dir / f"J_grid_used_N{N}.txt", "w") as f:
            for j in J_grid:
                f.write(f"{float(j):.12g}\n")

        print(f"J_grid ({len(J_grid)} pts): {', '.join([f'{j:.3f}' for j in J_grid])}", flush=True)

        for tau_c in tau_c_list:
            print(f"tau_c={tau_c}", flush=True)

            for beta_E in beta_E_list:
                print(f"  beta_E={beta_E}", flush=True)

                for seed in seeds:
                    u = generate_ou_stimulus(
                        dt=dt,
                        T=T,
                        tau_c=float(tau_c),
                        sigma_u=float(config.get("stimulus", {}).get("sigma_u", 1.0)),
                        mu_u=float(config.get("stimulus", {}).get("mu_u", 0.0)),
                        seed=int(seed),
                    )
                    _, dt_eff = effective_sampling_dt(u, dt)

                    for j_idx, J in enumerate(J_grid):
                        key = (int(N), int(seed), float(tau_c), float(beta_E), float(beta_C), float(J))
                        if key in done_keys:
                            completed += 1
                            continue

                        if completed % max(1, int(args.progress_every)) == 0:
                            print(
                                f"[progress] {completed}/{total_pts}  "
                                f"N={N} seed={seed} tau_c={tau_c} beta_E={beta_E} "
                                f"J={J:.4f} ({j_idx+1}/{len(J_grid)})",
                                flush=True,
                            )

                        error_msg = ""
                        try:
                            res = optimize_theta(
                                N=int(N),
                                dt=float(dt),
                                T=float(T),
                                u=u,
                                dt_eff=float(dt_eff),
                                J=float(J),
                                beta_E=float(beta_E),
                                beta_C=float(beta_C),
                                n_restarts=int(n_restarts),
                                seed=int(seed),
                                maxiter=int(maxiter_val),
                            )
                            diag = res.get("diagnostics", {}) or {}
                        except Exception as e:
                            res = {
                                "objective": np.nan,
                                "best_params": [np.nan, np.nan, np.nan],
                                "diagnostics": {"error": repr(e)},
                            }
                            diag = res["diagnostics"]
                            error_msg = repr(e)

                        best_params = res.get("best_params", [np.nan, np.nan, np.nan])
                        theta0, thetaV, thetaA = (best_params + [np.nan, np.nan, np.nan])[:3]

                        stable_flag = 1 if ("error" not in diag) else 0
                        I_dec = float(diag.get("i_dec", np.nan)) if stable_flag else np.nan
                        mean_rate = float(diag.get("mean_rate", np.nan)) if stable_flag else np.nan
                        objective = float(res.get("objective", np.nan)) if stable_flag else np.nan

                        Delta = float(Jc_used - J) if np.isfinite(Jc_used) else np.nan
                        if "error" in diag and not error_msg:
                            error_msg = str(diag.get("error"))

                        row = {
                            "N": int(N),
                            "seed": int(seed),
                            "tau_c": float(tau_c),
                            "beta_E": float(beta_E),
                            "beta_C": float(beta_C),
                            "J": float(J),
                            "Jc_used": float(Jc_used) if np.isfinite(Jc_used) else np.nan,
                            "Delta": float(Delta) if np.isfinite(Delta) else np.nan,
                            "theta0": float(theta0) if np.isfinite(theta0) else np.nan,
                            "thetaV": float(thetaV) if np.isfinite(thetaV) else np.nan,
                            "thetaA": float(thetaA) if np.isfinite(thetaA) else np.nan,
                            "I_dec": float(I_dec) if np.isfinite(I_dec) else np.nan,
                            "mean_rate": float(mean_rate) if np.isfinite(mean_rate) else np.nan,
                            "objective": float(objective) if np.isfinite(objective) else np.nan,
                            "stable_flag": int(stable_flag),
                            "error_msg": error_msg,
                        }

                        _append_row_csv(opt_rows_csv, row, fieldnames)
                        done_keys.add(key)
                        completed += 1


    if not opt_rows_csv.exists():
        print("ERROR: opt_rows.csv not created. Something went wrong.")
        return

    df_rows = pd.read_csv(opt_rows_csv)
    if len(df_rows) == 0:
        print("ERROR: opt_rows.csv is empty.")
        return

    df_summary_source = _apply_summary_filter(df_rows, args.rate_cap)

    summary_data: List[Dict[str, Any]] = []
    groups = df_summary_source.groupby(["N", "tau_c", "beta_E", "beta_C"])
    for (N, tau_c, beta_E, beta_C), group in groups:
        j_stats = (
            group.groupby("J")
            .agg(
                objective_mean=("objective", "mean"),
                objective_std=("objective", "std"),
                I_dec_mean=("I_dec", "mean"),
                rate_mean=("mean_rate", "mean"),
                n_valid=("objective", "size"),
            )
            .reset_index()
        )

        valid_j = j_stats[j_stats["n_valid"] > 0]
        if len(valid_j) > 0 and valid_j["objective_mean"].notna().any():
            best = valid_j.loc[valid_j["objective_mean"].idxmax()]
            J_star = float(best["J"])
            obj_star_mean = float(best["objective_mean"])
            obj_star_std = float(best["objective_std"]) if pd.notna(best["objective_std"]) else 0.0
            i_dec_star = float(best["I_dec_mean"]) if pd.notna(best["I_dec_mean"]) else np.nan
            rate_star = float(best["rate_mean"]) if pd.notna(best["rate_mean"]) else np.nan
            n_valid_points = int(best["n_valid"])
        else:
            J_star = np.nan
            obj_star_mean = np.nan
            obj_star_std = np.nan
            i_dec_star = np.nan
            rate_star = np.nan
            n_valid_points = 0

        Jc_val = group["Jc_used"].dropna().iloc[0] if group["Jc_used"].notna().any() else np.nan
        Delta_star = float(Jc_val - J_star) if (np.isfinite(Jc_val) and np.isfinite(J_star)) else np.nan
        Delta_star_norm = float(Delta_star / Jc_val) if (np.isfinite(Delta_star) and np.isfinite(Jc_val) and Jc_val != 0.0) else np.nan

        summary_data.append(
            {
                "N": int(N),
                "tau_c": float(tau_c),
                "beta_E": float(beta_E),
                "beta_C": float(beta_C),
                "Jc_used": float(Jc_val) if np.isfinite(Jc_val) else np.nan,
                "J_star": float(J_star) if np.isfinite(J_star) else np.nan,
                "Delta_star": float(Delta_star) if np.isfinite(Delta_star) else np.nan,
                "Delta_star_norm": float(Delta_star_norm) if np.isfinite(Delta_star_norm) else np.nan,
                "objective_star_mean": float(obj_star_mean) if np.isfinite(obj_star_mean) else np.nan,
                "objective_star_std": float(obj_star_std) if np.isfinite(obj_star_std) else np.nan,
                "I_dec_star_mean": float(i_dec_star) if np.isfinite(i_dec_star) else np.nan,
                "rate_star_mean": float(rate_star) if np.isfinite(rate_star) else np.nan,
                "n_valid_points": int(n_valid_points),
                "rate_cap_used": float(args.rate_cap) if args.rate_cap is not None else np.nan,
            }
        )

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(opt_summary_csv, index=False)

    safe_save_run_manifest(
        output_dir=out_dir,
        config=config,
        seeds={"seeds": seeds, "args": vars(args)},
        output_paths=[str(opt_rows_csv), str(opt_summary_csv)],
    )

    plot_prl_figures(df_rows, df_summary, out_dir, args.rate_cap)

    print("\n" + "=" * 80)
    print(f"Saved: {opt_rows_csv}")
    print(f"Saved: {opt_summary_csv}")
    print(f"Figures: {out_dir}/*.png")
    print("=" * 80)

    if int(df_rows["stable_flag"].sum()) == 0:
        print("CRITICAL WARNING: No stable points found at all!")
    if df_rows["objective"].isna().all():
        print("CRITICAL WARNING: Objective is everywhere NaN!")


if __name__ == "__main__":
    main()
