import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
sys.path.append(str(BASE_DIR))

from src.stimulus import generate_ou_stimulus
from src.simulate_mc import run_simulation
from src.info_estimators import estimate_information_decoder

REV_V3_DIR = BASE_DIR / "analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"

def run_tau_fast():
    N = 5000
    stab_csv = BASE_DIR / "tables" / "tuned_branch_stability_clean.csv"
    if not stab_csv.exists(): return
    df = pd.read_csv(stab_csv)
    df = df[(df['N'] == N) & (df['beta_E'] == 0.0)]
    
    J_vals = sorted(df['J'].unique())
    
    dt = 0.0005
    T = 40.0
    dt_eff = 0.005
    stride = int(dt_eff / dt)
    
    tau_vals = [0.02, 0.05, 0.10]
    seeds = [1, 2]
    
    results = []
    agg_plots = {tau: [] for tau in tau_vals}
    
    for J in J_vals:
        row = df[df['J'] == J].iloc[0]
        theta0, thetaA, thetaV = row['theta0'], row['thetaA'], row['thetaV']
        
        for tau in tau_vals:
            objs = []
            for seed in seeds:
                u_base = generate_ou_stimulus(dt, T, tau_c=tau, sigma_u=1.0, mu_u=0.0, seed=seed)
                res = run_simulation(
                    N=N, dt=dt, T=T, u=u_base, J=J,
                    lambda0=100.0, theta0=theta0, thetaV=thetaV, thetaA=thetaA,
                    tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
                    v_reset_val=0.0, noise_sigma=0.5, seed=seed
                )
                A_t = res['A_t']
                valid_mask = ~np.isnan(A_t)
                if not valid_mask.any(): continue
                
                A_t_eff = A_t[::stride]
                u_t_eff = u_base[::stride]
                
                idec, _, _ = estimate_information_decoder(A_t_eff, u_t_eff, dt_eff, lags=20, lag_step=1)
                objs.append(idec)
            
            mean_obj = np.mean(objs) if objs else np.nan
            results.append({
                'J': J,
                'tau_c': tau,
                'objective': mean_obj
            })
            
    res_df = pd.DataFrame(results)
    
    for tau in tau_vals:
        sub = res_df[res_df['tau_c'] == tau]
        agg_plots[tau] = sub
        
    final_res = []
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {0.02: 'blue', 0.05: 'black', 0.10: 'red'}
    for tau, sub in agg_plots.items():
        if len(sub) == 0: continue
        ax.plot(sub['J'], sub['objective'], marker='o', color=colors[tau], label=f'$\\tau_c$ = {tau}s')
        idx_max = sub['objective'].idxmax()
        j_star = sub.loc[idx_max, 'J']
        ax.axvline(j_star, color=colors[tau], linestyle='--')
        final_res.append({'tau_c': tau, 'J_star': j_star})
        
    ax.set_xlabel('Coupling J')
    ax.set_ylabel('Objective (bits/s)')
    ax.set_title(f'Input Timescale Robustness (N={N})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "FigR_input_stats_robustness.pdf")
    plt.savefig(FIG_DIR / "FigR_input_stats_robustness.png", dpi=300)
    plt.close()
    
    pd.DataFrame(final_res).to_csv(TBL_DIR / "input_stats_robustness.csv", index=False)
    print("Saved input stats fast evaluation.")

if __name__ == "__main__":
    run_tau_fast()
