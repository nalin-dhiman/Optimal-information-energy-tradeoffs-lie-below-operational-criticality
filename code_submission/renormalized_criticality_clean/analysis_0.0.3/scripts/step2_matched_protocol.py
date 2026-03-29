import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

BASE_DIR = Path()
sys.path.append(str(BASE_DIR))

from src.stimulus import generate_ou_stimulus
from src.simulate_mc import run_simulation
from src.criticality import estimate_susceptibility, estimate_correlation_time, check_stability

REV_V3_DIR = BASE_DIR / "revision_analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"


def run_matched_protocol():
    
    Ns = [2000, 5000]
    

    J_vals = np.linspace(0.40, 0.70, 31)
    

    tau_c = 0.05
    sigma_u = 1.0
    mu_u = 0.0
    eps = 0.02
    f0 = 5.0
    
    results = []
    
    for N in Ns:

        dt_hf = 0.0001
        T_hf = 120.0
        burn_in_hf = 20.0
        

        dt_opt = 0.0005
        T_opt = 10.0
        burn_in_opt = 2.0
        

        u_base_hf = generate_ou_stimulus(dt_hf, T_hf, tau_c, sigma_u, mu_u, seed=42)
        t_hf = np.arange(len(u_base_hf)) * dt_hf
        u_stim_hf = u_base_hf + eps * np.sin(2 * np.pi * f0 * t_hf)
        
        u_base_opt = generate_ou_stimulus(dt_opt, T_opt, tau_c, sigma_u, mu_u, seed=42)
        t_opt = np.arange(len(u_base_opt)) * dt_opt
        u_stim_opt = u_base_opt + eps * np.sin(2 * np.pi * f0 * t_opt)
        
        chi_hf = []
        chi_opt = []
        
        print(f"Testing N={N}...")
        for J in J_vals:

            res_hf = run_simulation(
                N=N, dt=dt_hf, T=T_hf, u=u_stim_hf, J=J,
                lambda0=100.0, theta0=-3.0, thetaV=1.0, thetaA=-0.5,
                tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
                v_reset_val=0.0, noise_sigma=0.5, seed=42
            )
            is_st_hf, _ = check_stability(res_hf['A_t'], dt_hf, burn_in_hf, r_max=200.0)
            if is_st_hf:
                c_hf = estimate_susceptibility(res_hf['A_t'], dt_hf, f0, eps, burn_in_hf)
            else:
                c_hf = np.nan
            chi_hf.append(c_hf)
            

            res_opt = run_simulation(
                N=N, dt=dt_opt, T=T_opt, u=u_stim_opt, J=J,
                lambda0=100.0, theta0=-3.0, thetaV=1.0, thetaA=-0.5,
                tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
                v_reset_val=0.0, noise_sigma=0.5, seed=42
            )
            is_st_opt, _ = check_stability(res_opt['A_t'], dt_opt, burn_in_opt, r_max=200.0)
            if is_st_opt:
                c_opt = estimate_susceptibility(res_opt['A_t'], dt_opt, f0, eps, burn_in_opt)
            else:
                c_opt = np.nan
            chi_opt.append(c_opt)
            
        chi_hf = np.array(chi_hf)
        chi_opt = np.array(chi_opt)
        
        valid_hf = np.where(~np.isnan(chi_hf))[0]
        valid_opt = np.where(~np.isnan(chi_opt))[0]
        
        Jc_hf = J_vals[valid_hf[np.argmax(chi_hf[valid_hf])]] if len(valid_hf) > 0 else np.nan
        Jc_opt = J_vals[valid_opt[np.argmax(chi_opt[valid_opt])]] if len(valid_opt) > 0 else np.nan
        
      
        j_star = np.nan
        csv_path = TBL_DIR / "uncertainty_summary_quantities.csv"
        if csv_path.exists():
            u_df = pd.read_csv(csv_path)
            u_sub = u_df[(u_df['N'] == N) & (u_df['beta_E'] == 0.0)]
            if not u_sub.empty:
                j_star = u_sub.iloc[0]['J_star_mean']
        
        results.append({
            'N': N,
            'Jc_Original_HighFidelity': Jc_hf,
            'Jc_Matched_Optimization': Jc_opt,
            'J_star': j_star,
            'Delta_HF': (Jc_hf - j_star)/Jc_hf if pd.notnull(j_star) and Jc_hf > 0 else np.nan,
            'Delta_Matched': (Jc_opt - j_star)/Jc_opt if pd.notnull(j_star) and Jc_opt > 0 else np.nan
        })
        

        fig, ax = plt.subplots(figsize=(6, 4))
        max_hf = np.nanmax(chi_hf) if not np.all(np.isnan(chi_hf)) else 1.0
        max_opt = np.nanmax(chi_opt) if not np.all(np.isnan(chi_opt)) else 1.0
        
        ax.plot(J_vals, chi_hf / max_hf, 'k-o', label='High-Fidelity (Manuscript)')
        ax.plot(J_vals, chi_opt / max_opt, 'r--s', label='Matched (Optimization)')
        ax.axvline(Jc_hf, color='k', linestyle=':', label=f'$J_c$ (HF) = {Jc_hf:.2f}')
        ax.axvline(Jc_opt, color='r', linestyle=':', label=f'$J_c$ (Matched) = {Jc_opt:.2f}')
        if pd.notnull(j_star):
            ax.axvline(j_star, color='b', linestyle='-', label=f'$J^*$ = {j_star:.2f}')
            
        ax.set_xlabel('Coupling J')
        ax.set_ylabel(r'Normalized $\chi$')
        ax.set_title(f'Protocol Mismatch Test (N={N})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"FigR_matched_protocol_control_N{N}.pdf")
        plt.close()
        
    out_df = pd.DataFrame(results)
    out_df.to_csv(TBL_DIR / "matched_protocol_markers.csv", index=False)
    print("Saved matched_protocol_markers.csv")

if __name__ == "__main__":
    run_matched_protocol()
