import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("")
sys.path.append(str(BASE_DIR))

from src.stimulus import generate_ou_stimulus
from src.simulate_mc import run_simulation
from src.info_estimators import estimate_information_decoder

REV_V3_DIR = BASE_DIR / "analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"


def run_supracritical_probe():
    N = 2000
    stab_csv = BASE_DIR /  "tables" / "tuned_branch_stability_clean.csv"
    if not stab_csv.exists(): return
    df = pd.read_csv(stab_csv)
    df = df[(df['N'] == N) & (df['beta_E'] == 0.0)]
    

    row = df.iloc[-1]
    J_max_valid = row['J']
    theta0 = row['theta0']
    thetaA = row['thetaA']
    thetaV = row['thetaV']
    Jc = row['Jc']
    
    dt = 0.0005
    T = 40.0
    dt_eff = 0.005
    stride = int(dt_eff / dt)
    seed = 42
    
    J_sweep = np.linspace(J_max_valid, 0.65, 11)
    
    results = []
    
    u_base = generate_ou_stimulus(dt, T, tau_c=0.05, sigma_u=1.0, mu_u=0.0, seed=seed)
    u_t_eff = u_base[::stride]
    
    for J in J_sweep:
        res = run_simulation(
            N=N, dt=dt, T=T, u=u_base, J=J,
            lambda0=100.0, theta0=theta0, thetaV=thetaV, thetaA=thetaA,
            tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
            v_reset_val=0.0, noise_sigma=0.5, seed=seed
        )
        
        A_t = res['A_t']
        

        mean_rate = np.nanmean(A_t)
        
        if np.isnan(A_t).any() or mean_rate > 100.0:
            idec = 0.0
            status = 'Diverged'
        else:
            A_t_eff = A_t[::stride]
            idec, _, _ = estimate_information_decoder(A_t_eff, u_t_eff, dt_eff, lags=20, lag_step=1)
            status = 'Stable'
            
        results.append({
            'J': J,
            'I_dec': idec,
            'mean_rate': mean_rate,
            'status': status
        })
        print(f"J={J:.3f} | Rate={mean_rate:.1f} | I_dec={idec:.2f} | {status}")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(TBL_DIR / "supracritical_probe.csv", index=False)
    
    fig, ax1 = plt.subplots(figsize=(6, 4))
    
    color = 'tab:blue'
    ax1.set_xlabel('Coupling J')
    ax1.set_ylabel('Decoder Information (bits/s)', color=color)
    p_valid = res_df[res_df['status'] == 'Stable']
    p_div = res_df[res_df['status'] == 'Diverged']
    ax1.plot(p_valid['J'], p_valid['I_dec'], 'o-', color=color)
    ax1.plot(p_div['J'], np.zeros(len(p_div)), 'x', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Mean Firing Rate (Hz)', color=color)  
    ax2.plot(res_df['J'], res_df['mean_rate'], 's--', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.axvline(Jc, color='gray', linestyle=':', label=f'$J_c \\approx {Jc:.2f}$')
    ax1.axvline(J_max_valid, color='k', linestyle='--', label=f'Optimized Limit ({J_max_valid:.2f})')
    
    fig.tight_layout()
    plt.title(f'Supra-Critical Extrapolation (N={N})')
    plt.savefig(FIG_DIR / "FigR_supracritical_probe.pdf")
    plt.savefig(FIG_DIR / "FigR_supracritical_probe.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_supracritical_probe()
