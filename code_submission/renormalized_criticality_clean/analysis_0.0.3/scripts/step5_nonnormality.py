import os
import sys
import numpy as np
import pandas as pd
import scipy.linalg as la
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

def compute_stability_matrix(A0, J, thetaA, thetaV, tau_s=0.02, tau_v=0.01):
    M11 = (A0 * thetaA - 1.0) / tau_s
    M12 = (A0 * thetaV) / tau_s
    M21 = J / tau_v
    M22 = -1.0 / tau_v
    return np.array([[M11, M12], [M21, M22]])

def analyze_nonnormality():
    results = []
    
    for opt_dir in OPT_DIRS:
        if not opt_dir.exists(): continue
        df = pd.read_csv(opt_dir / "opt_rows.csv")
        

        df = df[(df['stable_flag'] == 1) & (df['beta_E'] == 0.0)]
        
        for _, row in df.iterrows():
            N = row['N']
            J = row['J']
            Jc = row.get('Jc_used', np.nan)
            A0 = row['mean_rate']
            thetaA = row['thetaA']
            thetaV = row['thetaV']
            
            M = compute_stability_matrix(A0, J, thetaA, thetaV)

            eigenvalues, P = np.linalg.eig(M)
            spectral_abscissa = np.max(np.real(eigenvalues))
            


            try:
                kappa_P = np.linalg.cond(P)
            except np.linalg.LinAlgError:
                kappa_P = np.nan
                

            T, Z = la.schur(M, output='real')
            
            N_mat = np.triu(T, 1)
            henrici_dep = np.linalg.norm(N_mat, ord='fro')
            

            M_sym = 0.5 * (M + M.T)
            num_abscissa = np.max(np.linalg.eigvalsh(M_sym))
            
            results.append({
                'N': N,
                'J': J,
                'Jc': Jc,
                'mean_rate': A0,
                'spectral_abscissa': spectral_abscissa,
                'kappa_P': kappa_P,
                'henrici_departure': henrici_dep,
                'numerical_abscissa': num_abscissa
            })
            
    if not results:
        print("No valid data for non-normality compute.")
        return
        
    res_df = pd.DataFrame(results)
    

    agg_df = res_df.groupby(['N', 'J']).mean().reset_index()
    agg_df.to_csv(TBL_DIR / "nonnormality_summary.csv", index=False)
    print("Saved nonnormality_summary.csv")
    

    for N in agg_df['N'].unique():
        parent_df = agg_df[agg_df['N'] == N]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        

        axes[0].plot(parent_df['J'], parent_df['spectral_abscissa'], 'k-o', label='Spectral Abscissa (Re $\lambda$)')
        axes[0].plot(parent_df['J'], parent_df['numerical_abscissa'], 'r--s', label='Numerical Abscissa')
        axes[0].axhline(0, color='gray', linestyle=':')
        axes[0].set_xlabel('Coupling J')
        axes[0].set_ylabel('Abscissa')
        axes[0].set_title(f'N={N} Abscissas')
        axes[0].legend()
        if pd.notna(parent_df['Jc'].iloc[0]):
            axes[0].axvline(parent_df['Jc'].iloc[0], color='red', linestyle=':')
            

        axes[1].plot(parent_df['J'], parent_df['henrici_departure'], 'b-o')
        axes[1].set_xlabel('Coupling J')
        axes[1].set_ylabel('Norm of Upper Schur')
        axes[1].set_title(f'N={N} Henrici Departure')
        

        axes[2].plot(parent_df['J'], parent_df['kappa_P'], 'g-o')
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Coupling J')
        axes[2].set_ylabel('Condition Number $\kappa(P)$')
        axes[2].set_title(f'N={N} Eigenvector Cond')


        axes[3].plot(parent_df['J'], parent_df['mean_rate'], 'k-^')
        axes[3].set_xlabel('Coupling J')
        axes[3].set_ylabel('Mean Rate A0')
        axes[3].set_title(f'N={N} Rate Profile')

        plt.tight_layout()
        plt.savefig(FIG_DIR / f"FigR_nonnormality_vs_J_N{N}.pdf")
        plt.savefig(FIG_DIR / f"FigR_nonnormality_vs_J_N{N}.png", dpi=300)
        plt.close()

if __name__ == "__main__":
    analyze_nonnormality()
