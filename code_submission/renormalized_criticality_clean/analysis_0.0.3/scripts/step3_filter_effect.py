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

OPT_DIRS = [
    BASE_DIR / "results" / "opt_N2000_plateau_refine_20260318_115014",
    BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439",
    BASE_DIR / "results" / "opt_N8000_scale_20260319_150812",
    BASE_DIR / "results" / "opt_N10000_scale_20260319_150818",
    BASE_DIR / "results" / "opt_N12000_scale_20260319_150824"
]

def analyze_filter_effect():
    all_data = []
    
    for opt_dir in OPT_DIRS:
        opt_rows_path = opt_dir / "opt_rows.csv"
        if opt_rows_path.exists():
            df = pd.read_csv(opt_rows_path)
            all_data.append(df)
            
    if not all_data:
        print("No optimization data found.")
        return
        
    full_df = pd.concat(all_data, ignore_index=True)
    

    df_A = full_df[(full_df['stable_flag'] == 1) & (np.isfinite(full_df['objective']))].copy()
    

    df_B = df_A[(df_A['mean_rate'] < 20.0)].copy()
    
    results = []
    
    for N in full_df['N'].unique():
        for beta in [0.0, 0.2]:
            sub_A = df_A[(df_A['N'] == N) & (df_A['beta_E'] == beta)]
            sub_B = df_B[(df_B['N'] == N) & (df_B['beta_E'] == beta)]
            
            if len(sub_A) == 0: continue
            

            agg_A = sub_A.groupby('J')['objective'].mean().reset_index()
            agg_B = sub_B.groupby('J')['objective'].mean().reset_index()
            
            j_star_A = agg_A.loc[agg_A['objective'].idxmax(), 'J']
            j_star_B = agg_B.loc[agg_B['objective'].idxmax(), 'J'] if len(agg_B) > 0 else np.nan
            

            jc_used = sub_A['Jc_used'].median()
            

            near_jc_mask = sub_A['J'] >= jc_used - 0.05
            total_near = len(sub_A[near_jc_mask])
            kept_near = len(sub_B[sub_B['J'] >= jc_used - 0.05])
            excluded_near = total_near - kept_near
            pct_excluded = excluded_near / total_near if total_near > 0 else 0
            
            results.append({
                'N': N,
                'beta_E': beta,
                'J_star_Unfiltered': j_star_A,
                'J_star_Filtered': j_star_B,
                'J_*_Changed': j_star_A != j_star_B,
                'Total_Points_Near_Jc': total_near,
                'Excluded_Near_Jc': excluded_near,
                'Percent_Excluded_Near_Jc': pct_excluded
            })
            
            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(agg_A['J'], agg_A['objective'], 'k-o', label='Unfiltered (Stable Only)')
            if len(agg_B) > 0:
                ax.plot(agg_B['J'], agg_B['objective'], 'r--x', label=r'Filtered ($rate < 20$ Hz)')
                
            ax.axvline(jc_used, color='gray', linestyle=':', label='$J_c$')
            ax.axvline(j_star_A, color='k', linestyle='--', label=f'$J^*_A$ = {j_star_A:.3f}')
            if pd.notna(j_star_B) and j_star_A != j_star_B:
                ax.axvline(j_star_B, color='r', linestyle='--', label=f'$J^*_B$ = {j_star_B:.3f}')
                
            ax.set_xlabel('Coupling J')
            ax.set_ylabel('Mean Objective (bits/s)')
            ax.set_title(f'Filter Effect (N={N}, $\\beta_E$={beta})')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(FIG_DIR / f'FigR_filter_effect_N{N}_beta{beta}.pdf')
            plt.savefig(FIG_DIR / f'FigR_filter_effect_N{N}_beta{beta}.png', dpi=300)
            plt.close()
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(TBL_DIR / "filter_effect_summary.csv", index=False)
    print("Saved filter_effect_summary.csv")

if __name__ == "__main__":
    analyze_filter_effect()
