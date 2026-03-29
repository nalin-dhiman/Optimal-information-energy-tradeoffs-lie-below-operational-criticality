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
from src.criticality import estimate_susceptibility, check_stability

REV_DIR = BASE_DIR / "revision_analysis_v2"
FIG_DIR = REV_DIR / "figures"
TBL_DIR = REV_DIR / "tables"

def evaluate_susceptibility_grid(J_vals, N, dt, T, u_base, f0, eps, burn_in_s):
    t = np.arange(len(u_base)) * dt
    u_probe = eps * np.sin(2 * np.pi * f0 * t)
    u_stim = u_base + u_probe
    
    chis = []
    
    for J in J_vals:
        print(f"  Simulating J={J:.3f}...")
        res = run_simulation(
            N=N, dt=dt, T=T, u=u_stim, J=J,
            lambda0=100.0, theta0=-3.0, thetaV=1.0, thetaA=-0.5,
            tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
            v_reset_val=0.0, noise_sigma=0.5, seed=42
        )
        A_t = res['A_t']
        is_stable, metrics = check_stability(A_t, dt, burn_in_s, r_max=200.0)
        
        chi = np.nan
        if is_stable:
            chi = estimate_susceptibility(A_t, dt, f0, eps, burn_in_s)
            
        chis.append(chi)
        
    return np.array(chis)

def main():
    N = 2000
    dt = 0.0001
    T = 40.0
    burn_in_s = 10.0
    

    u_base = generate_ou_stimulus(dt, T, tau_c=0.05, sigma_u=1.0, mu_u=0.0, seed=42)
    

    J_vals = np.linspace(0.55, 0.605, 12)
    

    frequencies = [1.0, 5.0, 15.0]
    eps_val = 0.02 
    
    results = []
    chi_curves_f = {}
    

    for f0 in frequencies:
        print(f"Testing f0 = {f0} Hz...")
        chis = evaluate_susceptibility_grid(J_vals, N, dt, T, u_base, f0, eps_val, burn_in_s)
        chi_curves_f[f0] = chis
        
        valid_idx = np.where(~np.isnan(chis))[0]
        if len(valid_idx) > 0:
            Jc = J_vals[valid_idx[np.argmax(chis[valid_idx])]]
        else:
            Jc = np.nan
            
        results.append({
            'parameter': 'frequency',
            'value': f0,
            'Jc_chi': Jc,
            'max_chi': np.nanmax(chis) if not np.all(np.isnan(chis)) else np.nan
        })


    amplitudes = [0.01, 0.02, 0.05]
    f0_default = 5.0
    chi_curves_eps = {}
    

    for eps in amplitudes:
        if eps == 0.02 and 5.0 in chi_curves_f:
            chi_curves_eps[eps] = chi_curves_f[5.0]
            continue
            
        print(f"Testing eps = {eps}...")
        chis = evaluate_susceptibility_grid(J_vals, N, dt, T, u_base, f0_default, eps, burn_in_s)
        chi_curves_eps[eps] = chis
        
        valid_idx = np.where(~np.isnan(chis))[0]
        if len(valid_idx) > 0:
            Jc = J_vals[valid_idx[np.argmax(chis[valid_idx])]]
        else:
            Jc = np.nan
            
        results.append({
            'parameter': 'amplitude',
            'value': eps,
            'Jc_chi': Jc,
            'max_chi': np.nanmax(chis) if not np.all(np.isnan(chis)) else np.nan
        })
        
    res_df = pd.DataFrame(results)
    csv_out = TBL_DIR / "jc_probe_sensitivity_fixed.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")
    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    

    for f0 in frequencies:
        chis = chi_curves_f[f0]
        max_c = np.nanmax(chis) if not np.all(np.isnan(chis)) else 1.0
        ax1.plot(J_vals, chis/max_c, 'o-', label=f'$f_0={f0}$ Hz')
        
    ax1.set_xlabel('Coupling J')
    ax1.set_ylabel(r'Normalized $\chi$')
    ax1.set_title(r'Frequency Sensitivity ($\epsilon=0.02$)')
    ax1.axvline(0.59, color='k', linestyle=':', label='Manuscript $J_c$')
    ax1.legend()
    

    for eps in amplitudes:
        chis = chi_curves_eps[eps]
        max_c = np.nanmax(chis) if not np.all(np.isnan(chis)) else 1.0
        ax2.plot(J_vals, chis/max_c, 's--', label=rf'$\epsilon={eps}$')
        
    ax2.set_xlabel('Coupling J')
    ax2.set_ylabel(r'Normalized $\chi$')
    ax2.set_title(r'Amplitude Sensitivity ($f_0=5.0$)')
    ax2.axvline(0.59, color='k', linestyle=':', label='Manuscript $J_c$')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "FigR2_jc_probe_sensitivity_fixed.pdf")
    plt.savefig(FIG_DIR / "FigR2_jc_probe_sensitivity_fixed.png", dpi=300)
    plt.close()
    print("Saved FigR2_jc_probe_sensitivity_fixed")

if __name__ == "__main__":
    main()
