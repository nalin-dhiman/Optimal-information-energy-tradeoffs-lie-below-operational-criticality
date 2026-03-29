import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
RESULTS_DIR = BASE_DIR / "results"
FIG_SUB_DIR = Path()

OPT_DIRS = [
    RESULTS_DIR / "opt_N2000_plateau_refine_20260318_115014",
    RESULTS_DIR / "opt_N5000_plateau_precision_20260321_192439",
    RESULTS_DIR / "opt_N10000_scale_20260319_150818",
]

def make_fig1():
    all_data = []
    
    for opt_dir in OPT_DIRS:
        opt_rows_path = opt_dir / "opt_rows.csv"
        if opt_rows_path.exists():
            df = pd.read_csv(opt_rows_path)
            df = df[df['stable_flag'] == 1]
            all_data.append(df)
            
    if not all_data:
        print("No optimization data found.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df['beta_E'] == 0.0]
    
    sizes = [2000, 5000, 10000]
    

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, N in enumerate(sizes):
        ax = axes[i]
        df_N = full_df[full_df['N'] == N]
        if df_N.empty: 
            continue
            
        grouped = df_N.groupby('J')
        agg = grouped['objective'].agg(['mean', 'std', 'count']).reset_index()
        
        jc = df_N['Jc_used'].median()
        

        max_obj = agg['mean'].max()
        plateau = agg[agg['mean'] >= 0.95 * max_obj]
        if not plateau.empty:
            p_min = plateau['J'].min()
            p_max = plateau['J'].max()
            ax.axvspan(p_min, p_max, color='lightblue', alpha=0.15)
            

        sem = agg['std'] / np.sqrt(agg['count'].replace(0, 1))
        ax.fill_between(agg['J'], agg['mean'] - sem, agg['mean'] + sem, color='black', alpha=0.15)
        

        lbl_obj = 'Objective' if i == 0 else ""
        ax.plot(agg['J'], agg['mean'], color='black', marker='o', linestyle='-', linewidth=2, markersize=5, label=lbl_obj)
        

        lbl_jc = r'$J_c$' if i == 0 else ""
        ax.axvline(x=jc, color='red', linestyle='--', linewidth=1.5, label=lbl_jc)
        

        ax.set_title(f"N = {N}")
        ax.set_xlabel(r"Coupling $J$")
        ax.margins(y=0.05)
        
        if i == 0:
            ax.set_ylabel("Objective (bits/s)")
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=False)
    
    
    plt.tight_layout()
    FIG_SUB_DIR.mkdir(parents=True, exist_ok=True)
    
    out_pdf = FIG_SUB_DIR / "Fig1_objective_landscapes_clean.pdf"
    out_png = FIG_SUB_DIR / "Fig1_objective_landscapes_clean.png"
    
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    plt.close()
    print("Saved clean figures to manuscript/figures")

if __name__ == "__main__":
    make_fig1()
