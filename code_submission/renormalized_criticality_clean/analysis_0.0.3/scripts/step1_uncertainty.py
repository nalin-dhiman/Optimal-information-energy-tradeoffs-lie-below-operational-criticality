import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path()
REV_V3_DIR = BASE_DIR / 
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"


OPT_DIRS = [
    BASE_DIR / "results" / "opt_N2000_plateau_refine_20260318_115014",
    BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439",
    BASE_DIR / "results" / "opt_N8000_scale_20260319_150812",
    BASE_DIR / "results" / "opt_N10000_scale_20260319_150818",
    BASE_DIR / "results" / "opt_N12000_scale_20260319_150824"
]

def analyze_uncertainty():
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
    
   
    grouped = full_df.groupby(['N', 'beta_E', 'J'])
    agg = grouped['objective'].agg(['mean', 'std', 'count']).reset_index()
    agg['std'] = agg['std'].fillna(0.0)
    agg['stderr'] = agg['std'] / np.sqrt(agg['count'])
    agg['ci_lower_95'] = agg['mean'] - 1.96 * agg['stderr']
    agg['ci_upper_95'] = agg['mean'] + 1.96 * agg['stderr']
    
    agg.to_csv(TBL_DIR / "uncertainty_main_figures.csv", index=False)
    print("Saved uncertainty_main_figures.csv")
    

    jstar_stats = []
    
    for N in full_df['N'].unique():
        for beta in [0.0, 0.2]:
            sub_df = full_df[(full_df['N'] == N) & (full_df['beta_E'] == beta)]
            if len(sub_df) == 0: continue
            

            seeds = sub_df['seed'].unique()
            if len(seeds) < 2:
                continue
                

            seed_jstars = []
            seed_plateau_mins = []
            
            for s in seeds:
                s_df = sub_df[sub_df['seed'] == s]
                if s_df.empty: continue

                idx_max = s_df['objective'].idxmax()
                max_obj = s_df.loc[idx_max, 'objective']
                j_star = s_df.loc[idx_max, 'J']
                

                plateau = s_df[s_df['objective'] >= 0.95 * max_obj]
                p_min = plateau['J'].min()
                
                seed_jstars.append(j_star)
                seed_plateau_mins.append(p_min)
                
            j_star_mean = np.mean(seed_jstars)
            j_star_std = np.std(seed_jstars, ddof=1) if len(seed_jstars) > 1 else 0.0
            
            p_mean = np.mean(seed_plateau_mins)
            p_std = np.std(seed_plateau_mins, ddof=1) if len(seed_plateau_mins) > 1 else 0.0
            

            jc = sub_df['Jc_used'].median()
            delta_star_mean = (jc - j_star_mean) / jc

            delta_star_std = j_star_std / jc
            
            jstar_stats.append({
                'N': N,
                'beta_E': beta,
                'n_seeds': len(seed_jstars),
                'J_star_mean': j_star_mean,
                'J_star_std': j_star_std,
                'Plateau_Min_mean': p_mean,
                'Plateau_Min_std': p_std,
                'Delta_star_mean': delta_star_mean,
                'Delta_star_std': delta_star_std
            })
            
    jstar_df = pd.DataFrame(jstar_stats)
    jstar_df.to_csv(TBL_DIR / "uncertainty_summary_quantities.csv", index=False)
    print("Saved uncertainty_summary_quantities.csv")
    

    beta_vals = [0.0, 0.2]
    for beta in beta_vals:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        subset = agg[agg['beta_E'] == beta]
        if subset.empty: continue
        
        for N in sorted(subset['N'].unique()):
            N_df = subset[subset['N'] == N]
            
            ax.plot(N_df['J'], N_df['mean'], marker='o', label=f'N={int(N)}')
            ax.fill_between(N_df['J'], N_df['ci_lower_95'], N_df['ci_upper_95'], alpha=0.2)
            
        ax.set_xlabel('Coupling J')
        ax.set_ylabel('Objective (bits/s)')
        ax.set_title(rf'Objective vs J ($\beta_E$={beta}) with 95% CI')
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"FigR_uncertainty_objective_curves_beta{beta}.pdf")
        plt.savefig(FIG_DIR / f"FigR_uncertainty_objective_curves_beta{beta}.png", dpi=300)
        plt.close()
        
    print("Saved uncertainty figures.")

if __name__ == "__main__":
    analyze_uncertainty()
