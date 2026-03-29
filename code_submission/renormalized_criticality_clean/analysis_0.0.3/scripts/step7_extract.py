import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
REV_V3_DIR = BASE_DIR / "revision_analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"

TAU_RESULTS_DIR = REV_V3_DIR / "results_tauC"

def analyze_tau_robustness():
    N = 5000
    tau_vals = [0.02, 0.10]
    

    baseline_dir = BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439"
    baseline_csv = baseline_dir / "opt_summary.csv"
    
    plots = {}
    
    if baseline_csv.exists():
        b_df = pd.read_csv(baseline_csv)
        b_df = b_df[(b_df['beta_E'] == 0.0)]
        if not b_df.empty:
            b_rows = pd.read_csv(baseline_dir / "opt_rows.csv")
            b_rows = b_rows[(b_rows['stable_flag'] == 1) & (b_rows['beta_E'] == 0.0)]
            b_agg = b_rows.groupby('J')['objective'].mean().reset_index()
            plots[0.05] = b_agg
            
    results = []
    if 0.05 in plots:
        idx_max = plots[0.05]['objective'].idxmax()
        results.append({
            'tau_c': 0.05,
            'J_star': plots[0.05].loc[idx_max, 'J']
        })
    
    for tau in tau_vals:
        res_dir = TAU_RESULTS_DIR / f"opt_N5000_tauC_{tau}"
        rows_csv = res_dir / "opt_rows.csv"
        if rows_csv.exists():
            df = pd.read_csv(rows_csv)
            df = df[(df['stable_flag'] == 1)]
            if len(df) == 0: continue
            
            agg = df.groupby('J')['objective'].mean().reset_index()
            plots[tau] = agg
            idx_max = agg['objective'].idxmax()
            
            results.append({
                'tau_c': tau,
                'J_star': agg.loc[idx_max, 'J']
            })
            
    if not results:
        print("No tau results found.")
        return

    res_df = pd.DataFrame(results).sort_values(by='tau_c')
    res_df.to_csv(TBL_DIR / "input_stats_robustness.csv", index=False)
    print("Saved input_stats_robustness.csv")
    
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {0.02: 'blue', 0.05: 'black', 0.10: 'red'}
    for tau, agg in plots.items():
        if tau in colors:
            c = colors[tau]
            ax.plot(agg['J'], agg['objective'], marker='o', color=c, label=f'$\\tau_c$ = {tau}s')
            j_star = agg.loc[agg['objective'].idxmax(), 'J']
            ax.axvline(j_star, color=c, linestyle='--')
            
    ax.set_xlabel('Coupling J')
    ax.set_ylabel('Objective (bits/s)')
    ax.set_title(f'Input Timescale Robustness (N={N})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "FigR_input_stats_robustness.pdf")
    plt.savefig(FIG_DIR / "FigR_input_stats_robustness.png", dpi=300)
    plt.close()
    
    print("Saved input_stats_robustness figures.")

if __name__ == "__main__":
    analyze_tau_robustness()
