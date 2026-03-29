import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
REV_V3_DIR = BASE_DIR / "revision_analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"


OPT_DIRS = [
    BASE_DIR / "results" / "opt_N2000_plateau_refine_20260318_115014",
    BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439",
    BASE_DIR / "results" / "opt_N10000_scale_20260319_150818"
]

def analyze_betaC():
    betaC_vals = [0.0, 0.005, 0.01]
    results = []
    
    for opt_dir in OPT_DIRS:
        opt_rows_path = opt_dir / "opt_rows.csv"
        if not opt_rows_path.exists(): continue
        
        df = pd.read_csv(opt_rows_path)

        df = df[(df['stable_flag'] == 1) & (df['beta_E'] == 0.0)].copy()
        if len(df) == 0: continue
        

        df['l1_norm'] = df['theta0'].abs() + df['thetaA'].abs() + df['thetaV'].abs()
        
        N = df.iloc[0]['N']
        Jc = df['Jc_used'].median()
        
        plots = {}
        for bc in betaC_vals:
            df[f'obj_{bc}'] = df['I_dec'] - (bc * df['l1_norm'])
            
            agg = df.groupby('J')[f'obj_{bc}'].mean().reset_index()
            plots[bc] = agg
            
            idx_max = agg[f'obj_{bc}'].idxmax()
            j_star = agg.loc[idx_max, 'J']
            
            results.append({
                'N': N,
                'beta_C': bc,
                'Jc': Jc,
                'J_star': j_star
            })
            
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = {0.0: 'blue', 0.005: 'black', 0.01: 'red'}
        for bc in betaC_vals:
            agg = plots[bc]
            c = colors[bc]
            ax.plot(agg['J'], agg[f'obj_{bc}'], marker='o', color=c, label=f'$\\beta_C$ = {bc}')
            
            j_star = agg.loc[agg[f'obj_{bc}'].idxmax(), 'J']
            ax.axvline(j_star, color=c, linestyle='--')
            
        ax.axvline(Jc, color='gray', linestyle=':', label='$J_c$')
        ax.set_xlabel('Coupling J')
        ax.set_ylabel('Objective (bits/s)')
        ax.set_title(f'$\\beta_C$ Sensitivity (N={N})')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"FigR_betaC_sensitivity_N{N}.pdf")
        plt.savefig(FIG_DIR / f"FigR_betaC_sensitivity_N{N}.png", dpi=300)
        plt.close()
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(TBL_DIR / "betaC_sensitivity.csv", index=False)
    print("Saved betaC sensitivity outputs.")

if __name__ == "__main__":
    analyze_betaC()
