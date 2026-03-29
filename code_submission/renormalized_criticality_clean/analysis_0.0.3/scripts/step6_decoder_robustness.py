import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from pathlib import Path

BASE_DIR = Path()
sys.path.append(str(BASE_DIR))

from src.stimulus import generate_ou_stimulus
from src.simulate_mc import run_simulation
from src.info_estimators import build_lagged_features

REV_V3_DIR = BASE_DIR / "revision_analysis_v3"
TBL_DIR = REV_V3_DIR / "tables"
FIG_DIR = REV_V3_DIR / "figures"


def custom_decoder_eval(A_t, u_t, dt_eff, lags=20, lag_step=1, model_type='ridge'):
    try:
        X = build_lagged_features(A_t, lags=lags, step=lag_step)
    except ValueError:
        return np.nan
        
    offset = lags * lag_step
    y = u_t[offset:]
    if len(y) != X.shape[0]:
        min_len = min(len(y), X.shape[0])
        X = X[:min_len]
        y = y[:min_len]

    kf = KFold(n_splits=5, shuffle=False)
    gap = min(int(0.1 / dt_eff), len(y)//10)
    
    mses = []
    vars_u = []
    
    for train_index, test_index in kf.split(X):
        test_start = test_index[0]
        test_end = test_index[-1]
        valid_mask = (train_index < test_start - gap) | (train_index > test_end + gap)
        valid_train = train_index[valid_mask]
        
        if len(valid_train) == 0: continue
            
        X_train, y_train = X[valid_train], y[valid_train]
        X_test, y_test = X[test_index], y[test_index]
        
        if model_type == 'ridge':
            model = Ridge(alpha=1.0)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mses.append(np.mean((y_test - y_pred)**2))
        vars_u.append(np.var(y_test))
        
    mean_mse = np.mean(mses)
    mean_var = np.mean(vars_u)
    
    if mean_mse > 0 and mean_var > mean_mse:
        bits_per_sample = 0.5 * np.log2(mean_var / mean_mse)
        return bits_per_sample * (1.0 / dt_eff)
    return 0.0

def run_decoder_robustness():
    N = 5000

    stab_csv = BASE_DIR  "tables" / "tuned_branch_stability_clean.csv"
    if not stab_csv.exists():
        print("Required param file missing.")
        return
        
    df = pd.read_csv(stab_csv)
    df = df[(df['N'] == N) & (df['beta_E'] == 0.0)]
    
    J_vals = sorted(df['J'].unique())
    print(f"Testing N=5000 robustness over J = {J_vals}")
    
    dt = 0.0005
    T = 40.0  
    dt_eff = 0.005 
    stride = int(dt_eff / dt)
    
    results = []
    

    seeds = [1, 2, 3]
    
    for J in J_vals:
        row = df[df['J'] == J].iloc[0]
        Jc = row['Jc']
        theta0, thetaA, thetaV = row['theta0'], row['thetaA'], row['thetaV']
        
        idecs_ridge = []
        idecs_ols = []
        
        for seed in seeds:
            u_base = generate_ou_stimulus(dt, T, tau_c=0.05, sigma_u=1.0, mu_u=0.0, seed=seed)
            res = run_simulation(
                N=N, dt=dt, T=T, u=u_base, J=J,
                lambda0=100.0, theta0=theta0, thetaV=thetaV, thetaA=thetaA,
                tau_ref=0.005, tau_s=0.02, tau_v=0.01, do_v_reset=True,
                v_reset_val=0.0, noise_sigma=0.5, seed=seed
            )
            A_t = res['A_t']
            valid_mask = ~np.isnan(A_t)
            if not valid_mask.any(): continue
            

            A_t_eff = A_t[::stride]
            u_t_eff = u_base[::stride]
            
            val_ridge = custom_decoder_eval(A_t_eff, u_t_eff, dt_eff, lags=20, lag_step=1, model_type='ridge')
            val_ols = custom_decoder_eval(A_t_eff, u_t_eff, dt_eff, lags=20, lag_step=1, model_type='ols')
            
            idecs_ridge.append(val_ridge)
            idecs_ols.append(val_ols)
            
        mean_r = np.mean(idecs_ridge) if idecs_ridge else np.nan
        mean_o = np.mean(idecs_ols) if idecs_ols else np.nan
        
        results.append({
            'N': N,
            'J': J,
            'Jc': Jc,
            'I_dec_Ridge': mean_r,
            'I_dec_OLS': mean_o
        })
        print(f"J={J:.3f} | Ridge: {mean_r:.2f} | OLS: {mean_o:.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(TBL_DIR / "decoder_robustness.csv", index=False)
    
    J_star_ridge = res_df.loc[res_df['I_dec_Ridge'].idxmax(), 'J']
    J_star_ols = res_df.loc[res_df['I_dec_OLS'].idxmax(), 'J']
    Jc = res_df['Jc'].iloc[0]
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res_df['J'], res_df['I_dec_Ridge'], 'k-o', label='Baseline (Ridge)')
    ax.plot(res_df['J'], res_df['I_dec_OLS'], 'r--s', label='Alternative (OLS Autoregressive)')
    ax.axvline(Jc, color='gray', linestyle=':', label='$J_c$')
    ax.axvline(J_star_ridge, color='k', linestyle='--', label=f'$J^*_{{\\rm Ridge}}$ = {J_star_ridge:.2f}')
    
    if J_star_ols != J_star_ridge:
        ax.axvline(J_star_ols, color='r', linestyle='--', label=f'$J^*_{{\\rm OLS}}$ = {J_star_ols:.2f}')
        
    ax.set_xlabel('Coupling J')
    ax.set_ylabel('Objective (bits/s)')
    ax.set_title(f'Decoder Robustness Check (N={N})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "FigR_decoder_robustness.pdf")
    plt.savefig(FIG_DIR / "FigR_decoder_robustness.png", dpi=300)
    plt.close()
    
    print("Saved decoder_robustness.csv and figures.")

if __name__ == "__main__":
    run_decoder_robustness()
