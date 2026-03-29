import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
REV_DIR = BASE_DIR / 
FIG_DIR = REV_DIR / "figures"
TBL_DIR = REV_DIR / "tables"

OPT_DIRS = [
    BASE_DIR / "results" / "opt_N2000_plateau_refine_20260318_115014",
    BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439",
]

def compute_stability_matrix(A0, J, thetaA, thetaV, tau_s=0.02, tau_v=0.01):
    M11 = (A0 * thetaA - 1.0) / tau_s
    M12 = (A0 * thetaV) / tau_s
    M21 = J / tau_v
    M22 = -1.0 / tau_v
    return np.array([[M11, M12], [M21, M22]])

def analyze_stability():
    results = []
    
    for opt_dir in OPT_DIRS:
        if not opt_dir.exists(): continue
        df = pd.read_csv(opt_dir / "opt_rows.csv")
        
        
        df = df[df['stable_flag'] == 1]
        
        for _, row in df.iterrows():
            N = row['N']
            J = row['J']
            Jc = row['Jc_used']
            A0 = row['mean_rate']
            thetaA = row['thetaA']
            thetaV = row['thetaV']
            theta0 = row['theta0']
            beta_E = row['beta_E']
            
            M = compute_stability_matrix(A0, J, thetaA, thetaV)
            eigenvalues = np.linalg.eigvals(M)
            max_real_eig = np.max(np.real(eigenvalues))
            
            is_subcritical = max_real_eig < 0
            spectral_gap = -max_real_eig if is_subcritical else 0.0
            
            results.append({
                'N': N,
                'J': J,
                'Jc': Jc,
                'beta_E': beta_E,
                'theta0': theta0,
                'thetaA': thetaA,
                'thetaV': thetaV,
                'mean_rate': A0,
                'leading_real_eig': max_real_eig,
                'spectral_gap': spectral_gap,
                'subcritical_flag': int(is_subcritical)
            })
            
    res_df = pd.DataFrame(results)
    csv_out = TBL_DIR / "tuned_branch_stability_clean.csv"
    res_df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")
    

    plot_df = res_df[res_df['beta_E'] == 0.0]
    agg_df = plot_df.groupby(['N', 'J']).mean().reset_index()
    
    for N in agg_df['N'].unique():
        parent_df = agg_df[agg_df['N'] == N]
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(parent_df['J'], parent_df['leading_real_eig'], 'o-', color='black', label='Leading eigenvalue')
        ax.axhline(0, color='gray', linestyle='--')
        
        Jc_val = parent_df['Jc'].iloc[0]
        ax.axvline(Jc_val, color='red', linestyle=':', label=r'$J_c$ (marker)')
        
        
        mech_csv = TBL_DIR / "mechanism_summary_fixed.csv"
        if mech_csv.exists():
            mech_df = pd.read_csv(mech_csv)
            mech_parent = mech_df[mech_df['N'] == N]
            if not mech_parent.empty:
                J_star = mech_parent.groupby('J')['L_recomputed'].mean().idxmax()
                ax.axvline(J_star, color='blue', linestyle='--', label=r'$J^*$ (optimum)')
                
        ax.set_xlabel('Coupling J')
        ax.set_ylabel(r'Re($\lambda_1$) of M')
        ax.set_title(f"Stability along Tuned Branch (N={N})")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"FigR3_tuned_branch_stability_clean_N{N}.pdf")
        plt.close()

if __name__ == "__main__":
    analyze_stability()
