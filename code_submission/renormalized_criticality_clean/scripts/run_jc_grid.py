import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_config, save_run_manifest
from src.stimulus import generate_ou_stimulus
from src.simulate_mc import run_simulation
from src.criticality import estimate_susceptibility, estimate_correlation_time, check_stability, select_critical_coupling

def evaluate_J_grid(J_vals, N, dt, T, u_base, config, seeds, pass_id):
    eps_probe = config['criticality']['eps_probe']
    f0_probe = config['criticality']['f0_probe']
    burn_in_s = config['criticality']['burn_in_s']
    
    t = np.arange(len(u_base)) * dt
    u_probe = eps_probe * np.sin(2 * np.pi * f0_probe * t)
    u_stim = u_base + u_probe
    
    rows = []
    
    for J in J_vals:
        for seed in seeds:
            res = run_simulation(
                N=N, dt=dt, T=T, u=u_stim, J=J,
                lambda0=config['simulation']['lambda0'],
                theta0=config['simulation']['theta0'],
                thetaV=config['simulation']['thetaV'],
                thetaA=config['simulation']['thetaA'],
                tau_ref=config['simulation']['tau_ref'],
                tau_s=config['simulation']['tau_s'],
                tau_v=config['simulation']['tau_v'],
                do_v_reset=config['simulation']['do_v_reset'],
                v_reset_val=config['simulation']['v_reset_val'],
                noise_sigma=config['simulation']['noise_sigma'],
                seed=seed
            )
            
            A_t = res['A_t']
            is_stable, metrics = check_stability(
                A_t, dt, burn_in_s, 
                r_max=config['criticality']['r_max']
            )
            
            chi = np.nan
            tau_int = np.nan
            tau_fit = np.nan
            
            if is_stable:
                chi = estimate_susceptibility(A_t, dt, f0_probe, eps_probe, burn_in_s)
                tau_fit, tau_int = estimate_correlation_time(A_t, dt, burn_in_s)
                
            rows.append({
                'N': N,
                'J': J,
                'seed': seed,
                'pass_id': pass_id,
                'chi': chi,
                'tau_int': tau_int,
                'tau_fit': tau_fit,
                'mean_rate': metrics['mean_rate'],
                'stable_flag': int(is_stable),
                'edge_peak_flag': 0,
                'linear_response_flag': 0,
                'burn_in_s': burn_in_s,
                'eps_probe': eps_probe,
                'f0_probe': f0_probe,
                'T': T,
                'dt': dt
            })
            
    df = pd.DataFrame(rows)
    return df

def aggregate_J_grid(df):
    agg = df.groupby('J').agg({
        'chi': ['mean', 'std'],
        'tau_int': ['mean', 'std'],
        'mean_rate': 'mean',
        'stable_flag': 'mean' 
    }).reset_index()
    
    agg.columns = ['J', 'chi_mean', 'chi_std', 'tau_int_mean', 'tau_int_std', 'mean_rate', 'stable_frac']
    

    stable_mask = agg['stable_frac'] == 1.0
    
    Jc_chi_agg, edge_peak_chi = select_critical_coupling(agg['J'].values, agg['chi_mean'].values, stable_mask.values)
    Jc_tau_agg, edge_peak_tau = select_critical_coupling(agg['J'].values, agg['tau_int_mean'].values, stable_mask.values)
    
    return agg, stable_mask.values, Jc_chi_agg, edge_peak_chi, Jc_tau_agg, edge_peak_tau

def get_per_seed_jc(df_full):
    seeds = df_full['seed'].unique()
    jc_chi_list = []
    jc_tau_list = []
    ep_chi_any = False
    ep_tau_any = False
    
    for s in seeds:
        df_s = df_full[df_full['seed'] == s].sort_values('J')
        stable_mask = df_s['stable_flag'].values
        J = df_s['J'].values
        chi = df_s['chi'].values
        tau = df_s['tau_int'].values
        
        jc_chi, ep_chi = select_critical_coupling(J, chi, stable_mask)
        jc_tau, ep_tau = select_critical_coupling(J, tau, stable_mask)
        
        ep_chi_any = ep_chi_any or ep_chi
        ep_tau_any = ep_tau_any or ep_tau
        
        if not np.isnan(jc_chi) and not ep_chi:
            jc_chi_list.append(jc_chi)
        if not np.isnan(jc_tau) and not ep_tau:
            jc_tau_list.append(jc_tau)
            
    Jc_chi_mean = np.mean(jc_chi_list) if len(jc_chi_list) > 0 else np.nan
    Jc_chi_std = np.std(jc_chi_list) if len(jc_chi_list) > 1 else 0.0
    Jc_tau_mean = np.mean(jc_tau_list) if len(jc_tau_list) > 0 else np.nan
    Jc_tau_std = np.std(jc_tau_list) if len(jc_tau_list) > 1 else 0.0
    
    return Jc_chi_mean, Jc_chi_std, Jc_tau_mean, Jc_tau_std, ep_chi_any, ep_tau_any

def plot_diagnostic(agg, stable_mask, Jc_chi, Jc_tau, N, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    J = agg['J'].values
    chi_m = agg['chi_mean'].values
    chi_s = agg['chi_std'].values
    tau_m = agg['tau_int_mean'].values
    tau_s = agg['tau_int_std'].values
    

    ax1.errorbar(J[stable_mask], chi_m[stable_mask], yerr=chi_s[stable_mask], fmt='o-', label='Stable')
    ax1.errorbar(J[~stable_mask], chi_m[~stable_mask], yerr=chi_s[~stable_mask], fmt='x', color='red', label='Unstable')
    if not np.isnan(Jc_chi):
        ax1.axvline(Jc_chi, color='k', linestyle='--', label=f'Jc_chi = {Jc_chi:.3f}')
    ax1.set_xlabel('J')
    ax1.set_ylabel(r'$\chi$')
    ax1.set_title(f'Susceptibility N={N}')
    ax1.legend()
    
     
    ax2.errorbar(J[stable_mask], tau_m[stable_mask], yerr=tau_s[stable_mask], fmt='s-', color='orange', label='Stable')
    ax2.errorbar(J[~stable_mask], tau_m[~stable_mask], yerr=tau_s[~stable_mask], fmt='x', color='red', label='Unstable')
    if not np.isnan(Jc_tau):
        ax2.axvline(Jc_tau, color='k', linestyle='--', label=f'Jc_tau = {Jc_tau:.3f}')
    ax2.set_xlabel('J')
    ax2.set_ylabel(r'$\tau_{int}$ (s)')
    ax2.set_title(f'Correlation Time N={N}')
    ax2.legend()
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

def linear_response_test(J_test, N, dt, T, u_base, config, seeds):

    f0_probe = config['criticality']['f0_probe']
    burn_in_s = config['criticality']['burn_in_s']
    t = np.arange(len(u_base)) * dt
    
    def get_chi_for_eps(eps_mult):
        eps = config['criticality']['eps_probe'] * eps_mult
        u_probe = eps * np.sin(2 * np.pi * f0_probe * t)
        u_stim = u_base + u_probe
        chis = []
        for seed in seeds:
            res = run_simulation(
                N=N, dt=dt, T=T, u=u_stim, J=J_test,
                lambda0=config['simulation']['lambda0'],
                theta0=config['simulation']['theta0'],
                thetaV=config['simulation']['thetaV'],
                thetaA=config['simulation']['thetaA'],
                tau_ref=config['simulation']['tau_ref'],
                tau_s=config['simulation']['tau_s'],
                tau_v=config['simulation']['tau_v'],
                do_v_reset=config['simulation']['do_v_reset'],
                v_reset_val=config['simulation']['v_reset_val'],
                noise_sigma=config['simulation']['noise_sigma'],
                seed=seed
            )
            A_t = res['A_t']
            chi = estimate_susceptibility(A_t, dt, f0_probe, eps, burn_in_s)
            chis.append(chi)
        return np.mean(chis)
        
    chi_half = get_chi_for_eps(0.5)
    chi_double = get_chi_for_eps(2.0)
    chi_base = get_chi_for_eps(1.0) 
    
    if chi_base == 0:
        return False
        
    err_half = abs(chi_half - chi_base) / chi_base
    err_double = abs(chi_double - chi_base) / chi_base
    
    return err_half < 0.1 and err_double < 0.1

def parse_list(s, t=int):
    if s is None:
        return None
    return [t(x.strip()) for x in s.split(',')]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--outdir", default=None, type=str)
    parser.add_argument("--N_list", default=None, type=str)
    parser.add_argument("--seed_list", default=None, type=str)
    parser.add_argument("--pass_mode", choices=["coarse", "refine", "both"], default="both")
    parser.add_argument("--refine_min", type=float, default=None)
    parser.add_argument("--refine_max", type=float, default=None)
    parser.add_argument("--coarse_T", type=float, default=None)
    parser.add_argument("--coarse_dt", type=float, default=None)
    parser.add_argument("--refine_T", type=float, default=None)
    parser.add_argument("--refine_dt", type=float, default=None)
    parser.add_argument("--smoke_test", action="store_true", help="Run a fast smoke test")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    N_list_cfg = [1000, 2000, 5000, 10000]
    seeds_cfg = config['sweep'].get('seeds', [1, 2, 3, 4, 5])
    
    N_list = parse_list(args.N_list) or N_list_cfg
    seeds = parse_list(args.seed_list) or seeds_cfg
    
    if args.smoke_test:

        N_list = [200]
        config['criticality']['burn_in_s'] = 1.0
        seeds = [1, 2]
    
    subset_mode = args.smoke_test or (args.N_list is not None) or (args.seed_list is not None)
    
    out_dir = Path(args.outdir) if args.outdir else Path("results/jc")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    debug_dir = Path("results/jc_debug")
    if not subset_mode:
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Jc grid over N={N_list}, pass_mode={args.pass_mode}", flush=True)
    
    all_raw_rows = []
    summary_rows = []
    
    for N in N_list:
        print(f"\nEvaluating N={N}...", flush=True)
        
        df_coarse = None
        

        if args.pass_mode in ['coarse', 'both']:
            print("  Pass 1: Coarse Grid...", flush=True)
            T_c = args.coarse_T if args.coarse_T else (5.0 if args.smoke_test else config['simulation']['T'])
            dt_c = args.coarse_dt if args.coarse_dt else (0.0005 if args.smoke_test else config['simulation']['dt'])
            
            J_coarse = np.linspace(0.0, 3.0, 31)
            if args.smoke_test:
                J_coarse = np.linspace(0.0, 2.5, 6)
                
            u_base_coarse = generate_ou_stimulus(
                dt=dt_c, T=T_c, tau_c=config['stimulus']['tau_c'], 
                sigma_u=config['stimulus']['sigma_u'], mu_u=config['stimulus']['mu_u'], 
                seed=42
            )
            df_coarse = evaluate_J_grid(J_coarse, N, dt_c, T_c, u_base_coarse, config, seeds, "coarse")
            all_raw_rows.append(df_coarse)


        if args.pass_mode in ['refine', 'both']:
            print("  Pass 2: Refined Grid...", flush=True)
            T_r = args.refine_T if args.refine_T else (5.0 if args.smoke_test else config['simulation']['T'])
            dt_r = args.refine_dt if args.refine_dt else (0.0005 if args.smoke_test else config['simulation']['dt'])
            
            if args.refine_min is not None and args.refine_max is not None:
                J_min = args.refine_min
                J_max = args.refine_max
            elif args.pass_mode == 'both' and df_coarse is not None:
                stable_mask_coarse = df_coarse.groupby('J')['stable_flag'].mean() == 1.0
                agg_coarse = df_coarse.groupby('J').mean(numeric_only=True).reset_index()
                Jc_chi_coarse, _ = select_critical_coupling(agg_coarse['J'].values, agg_coarse['chi'].values, stable_mask_coarse.values)
                Jc_tau_coarse, _ = select_critical_coupling(agg_coarse['J'].values, agg_coarse['tau_int'].values, stable_mask_coarse.values)
                
                valid_peaks = [j for j in [Jc_chi_coarse, Jc_tau_coarse] if not np.isnan(j)]
                if len(valid_peaks) > 0:
                    coarse_peak = np.mean(valid_peaks)
                else:
                    stable_js = agg_coarse[stable_mask_coarse.values]['J'].values
                    coarse_peak = stable_js[-1] - 0.5 if len(stable_js) > 0 else 1.0
                    
                J_min = max(0.0, coarse_peak - 0.3)
                J_max = coarse_peak + 0.3
            else:
                raise ValueError("Refine pass requires --refine_min/--refine_max or --pass_mode both")
                
            J_refined = np.linspace(J_min, J_max, 41)
            if args.smoke_test:
                J_refined = np.linspace(J_min, J_max, 5)
            
            print(f"    Grid: [{J_min:.2f}, {J_max:.2f}]", flush=True)
            u_base_refine = generate_ou_stimulus(
                dt=dt_r, T=T_r, tau_c=config['stimulus']['tau_c'], 
                sigma_u=config['stimulus']['sigma_u'], mu_u=config['stimulus']['mu_u'], 
                seed=42
            )
            df_refined = evaluate_J_grid(J_refined, N, dt_r, T_r, u_base_refine, config, seeds, "refine")
            all_raw_rows.append(df_refined)

    df_combined = pd.concat(all_raw_rows).drop_duplicates(subset=['N', 'J', 'seed', 'pass_id']).sort_values(['N', 'pass_id', 'J'])
    
    if subset_mode:
        for N in df_combined['N'].unique():
            for seed in df_combined['seed'].unique():
                df_sub = df_combined[(df_combined['N'] == N) & (df_combined['seed'] == seed)]
                if len(df_sub) == 0:
                    continue
                row_path = out_dir / f"rows_N{N}_seed{seed}.csv"
                df_sub.to_csv(row_path, index=False)
                manifest_path = out_dir / f"run_manifest_N{N}_seed{seed}.json"
                save_run_manifest(
                    output_dir=out_dir,
                    config=config,
                    seeds={'global': 42, 'sweep': [seed]},
                    output_paths=[str(row_path)],
                    filename=manifest_path.name
                )
                print(f"Saved {row_path}", flush=True)
    else:

        raw_path = out_dir / "jc_scaling_raw.csv"
        df_combined.to_csv(raw_path, index=False)
        
        for N in N_list:
            df_N = df_combined[df_combined['N'] == N]
            agg_fin, mask_fin, Jc_chi_agg, ep_chi_agg, Jc_tau_agg, ep_tau_agg = aggregate_J_grid(df_N)
            Jc_chi_mean, Jc_chi_std, Jc_tau_mean, Jc_tau_std, ep_chi_any, ep_tau_any = get_per_seed_jc(df_N)
            
            lr_flag = False
            if not np.isnan(Jc_chi_mean):
                print(f"  Testing linear response at Jc_chi = {Jc_chi_mean:.3f}...", flush=True)
                T_r = args.refine_T if args.refine_T else (5.0 if args.smoke_test else config['simulation']['T'])
                dt_r = args.refine_dt if args.refine_dt else (0.0005 if args.smoke_test else config['simulation']['dt'])
                u_base_refine = generate_ou_stimulus(
                    dt=dt_r, T=T_r, tau_c=config['stimulus']['tau_c'], 
                    sigma_u=config['stimulus']['sigma_u'], mu_u=config['stimulus']['mu_u'], 
                    seed=42
                )
                lr_flag = linear_response_test(Jc_chi_mean, N, dt_r, T_r, u_base_refine, config, seeds)
                
            print(f"  -> Jc_chi_mean: {Jc_chi_mean:.3f} +- {Jc_chi_std:.3f}")
            print(f"  -> Jc_tau_mean: {Jc_tau_mean:.3f} +- {Jc_tau_std:.3f}")
            
            plot_path = debug_dir / f"curves_N{N}.png"
            plot_diagnostic(agg_fin, mask_fin, Jc_chi_agg, Jc_tau_agg, N, plot_path)
            
            summary_rows.append({
                'N': N,
                'Jc_chi_mean': Jc_chi_mean,
                'Jc_chi_std': Jc_chi_std,
                'Jc_tau_mean': Jc_tau_mean,
                'Jc_tau_std': Jc_tau_std,
                'edge_peak_chi': ep_chi_any,
                'edge_peak_tau': ep_tau_any,
                'linear_response_flag': lr_flag
            })
            
        df_summary = pd.DataFrame(summary_rows)
        sum_path = out_dir / "jc_scaling_summary.csv"
        df_summary.to_csv(sum_path, index=False)
        
        save_run_manifest(
            output_dir=out_dir,
            config=config,
            seeds={'global': 42, 'sweep': seeds},
            output_paths=[str(raw_path), str(sum_path)]
        )
        
        print("\n" + "="*40, flush=True)
        print("SANITY CHECK RESULTS", flush=True)
        print("="*40, flush=True)
        print("Jc_ren summary:", flush=True)
        for _, row in df_summary.iterrows():
            print(f"N={int(row['N'])}: Jc_chi={row['Jc_chi_mean']:.3f}, Jc_tau={row['Jc_tau_mean']:.3f}, LR_flag={row['linear_response_flag']}", flush=True)
            
        Jc_chi_vals = df_summary['Jc_chi_mean'].dropna().values
        trend = "flat"
        if len(Jc_chi_vals) > 1:
            if np.all(np.diff(Jc_chi_vals) >= 0): trend = "increasing"
            elif np.all(np.diff(Jc_chi_vals) <= 0): trend = "decreasing"
            else: trend = "mixed"
        print(f"Trend: {trend}", flush=True)
        
        agreements = []
        for _, row in df_summary.iterrows():
            if not np.isnan(row['Jc_chi_mean']) and not np.isnan(row['Jc_tau_mean']):
                threshold = max(row['Jc_chi_std'], row['Jc_tau_std'], 0.05)
                diff = abs(row['Jc_chi_mean'] - row['Jc_tau_mean'])
                agreements.append(diff <= threshold)
                
        if all(agreements) and len(agreements) > 0:
            print("Agreement Jc_chi ~ Jc_tau: YES (within threshold)", flush=True)
        else:
            print("Agreement Jc_chi ~ Jc_tau: NO", flush=True)
            
        edge_triggered = df_summary['edge_peak_chi'].any() or df_summary['edge_peak_tau'].any()
        print(f"Edge peak flag triggered: {edge_triggered}", flush=True)
        
        confidence = "high"
        if trend == "mixed": confidence = "medium"
        if not all(agreements) and len(agreements) > 0: confidence = "medium"
        if edge_triggered: confidence = "low"
        print(f"Confidence: {confidence}", flush=True)

if __name__ == "__main__":
    main()
