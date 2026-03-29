import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json

BASE_DIR = Path()
sys.path.append(str(BASE_DIR))

from src.simulate_mc import run_simulation
from src.stimulus import generate_ou_stimulus, effective_sampling_dt
from src.info_estimators import build_lagged_features

REV_DIR = BASE_DIR / "revision_analysis_v2"
REV_DIR.mkdir(exist_ok=True)
(REV_DIR / "figures").mkdir(exist_ok=True)
(REV_DIR / "tables").mkdir(exist_ok=True)
(REV_DIR / "scripts").mkdir(exist_ok=True)

OPT_DIRS = [
    BASE_DIR / "results" / "opt_N2000_plateau_refine_20260318_115014",
    BASE_DIR / "results" / "opt_N5000_plateau_precision_20260321_192439",
    # BASE_DIR / "results" / "opt_N8000_scale_20260319_150812",
    # BASE_DIR / "results" / "opt_N10000_scale_20260319_150818",
]

def decompose_mse_kfold(A_t, u_t, dt_eff, dt):
    lags = 20
    lag_step = max(1, int(0.005 / dt))
    alpha = 1.0
    n_splits = 5

    try:
        X = build_lagged_features(A_t, lags=lags, step=lag_step)
    except ValueError:
        return 0.0, 0.0, 0.0, 0.0
        
    offset = lags * lag_step
    y = u_t[offset:]
    
    if len(y) != X.shape[0]:
        min_len = min(len(y), X.shape[0])
        X = X[:min_len]
        y = y[:min_len]

    kf = KFold(n_splits=n_splits, shuffle=False)
    gap = min(int(0.1 / dt_eff), len(y)//10)
    
    mses = []
    vars_u = []
    biases_sq = []
    var_errs = []
    
    for train_index, test_index in kf.split(X):
        test_start = test_index[0]
        test_end = test_index[-1]
        valid_mask = (train_index < test_start - gap) | (train_index > test_end + gap)
        valid_train = train_index[valid_mask]
        if len(valid_train) == 0: continue
        
        X_train, y_train = X[valid_train], y[valid_train]
        X_test, y_test = X[test_index], y[test_index]
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        var_y = np.var(y_test)
        
        mean_y = np.mean(y_test)
        mean_pred = np.mean(y_pred)
        bias_sq = (mean_pred - mean_y)**2
        var_err = np.var(y_test - y_pred)
        
        mses.append(mse)
        vars_u.append(var_y)
        biases_sq.append(bias_sq)
        var_errs.append(var_err)
        
    mean_mse = np.mean(mses)
    mean_var = np.mean(vars_u)
    mean_bias_sq = np.mean(biases_sq)
    mean_var_err = np.mean(var_errs)
    
    if mean_mse > 0 and mean_var > mean_mse:
        bits_per_sample = 0.5 * np.log2(mean_var / mean_mse)
    else:
        bits_per_sample = 0.0
    i_dec = bits_per_sample * (1.0 / dt_eff)
    
    return i_dec, mean_mse, mean_bias_sq, mean_var_err

def run_diagnostics():
    results = []
    

    sigma_u = 1.0
    mu_u = 0.0
    
    for opt_dir in OPT_DIRS:
        if not opt_dir.exists(): continue
        

        manifest_path = opt_dir / "run_manifest.json"
        

        dt = 0.0001
        T = 120.0
        
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            args = manifest.get('seeds', {}).get('args', {})
            config = manifest.get('config', {})
            
            dt = float(args.get('dt_opt', config.get('simulation', {}).get('dt', dt)))
            T = float(args.get('T_opt', config.get('simulation', {}).get('T', T)))
            
        print(f"Loaded {opt_dir.name} -> dt={dt}, T={T}")
        
        opt_rows = opt_dir / "opt_rows.csv"
        df = pd.read_csv(opt_rows)
        

        df = df[df['beta_E'] == 0.0]
        df = df[df['stable_flag'] == 1]
        
        for idx, row in df.iterrows():
            N = int(row['N'])
            seed = int(row['seed'])
            tau_c = float(row['tau_c'])
            beta_E = float(row['beta_E'])
            beta_C = float(row['beta_C'])
            J = float(row['J'])
            theta0 = float(row['theta0'])
            thetaV = float(row['thetaV'])
            thetaA = float(row['thetaA'])
            Jc = float(row['Jc_used'])
            

            u_t = generate_ou_stimulus(dt=dt, T=T, tau_c=tau_c, sigma_u=sigma_u, mu_u=mu_u, seed=seed)
            _, dt_eff = effective_sampling_dt(u_t, dt)
            

            sim_res = run_simulation(N=N, dt=dt, T=T, u=u_t, J=J, lambda0=100.0, 
                                     theta0=theta0, thetaV=thetaV, thetaA=thetaA, seed=seed)
            A_t = sim_res['A_t']
            mean_rate = sim_res['mean_rate']
            

            i_dec, mse, bias_sq, var_err = decompose_mse_kfold(A_t, u_t, dt_eff, dt)
            
            l1_norm = np.abs(theta0) + np.abs(thetaV) + np.abs(thetaA)
            L_recomputed = i_dec - beta_E * mean_rate - beta_C * l1_norm
            L_recorded = row['objective']
            
            results.append({
                'N': N, 'J': J, 'Jc': Jc, 'seed': seed,
                'L_recorded': L_recorded,
                'L_recomputed': L_recomputed,
                'i_dec': i_dec, 'mse': mse,
                'bias_sq': bias_sq, 'var_err': var_err,
                'mean_rate': mean_rate,
                'reg_cost': beta_C * l1_norm,
                'error_diff': abs(L_recorded - L_recomputed)
            })
            print(f"N={N} J={J:.3f} seed={seed} diff={abs(L_recorded - L_recomputed):.6f}")

    res_df = pd.DataFrame(results)
    out_csv = REV_DIR / "tables" / "mechanism_summary_fixed.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    

    for N in res_df['N'].unique():
        sub_df = res_df[res_df['N'] == N]
        

        agg = sub_df.groupby('J').mean().reset_index()
        Jc = agg['Jc'].iloc[0]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        

        axes[0].plot(agg['J'], agg['L_recorded'], 'k-o', label='Recorded L')
        axes[0].plot(agg['J'], agg['L_recomputed'], 'r--x', label='Recomputed L')
        axes[0].axvline(Jc, color='gray', linestyle='dotted', label='$J_c$')
        axes[0].set_xlabel('Coupling $J$')
        axes[0].set_ylabel('Objective')
        axes[0].set_title(f"Objective Reconstruction N={N}")
        axes[0].legend()
        

        axes[1].plot(agg['J'], agg['mse'], 'k-o', label='Total MSE')
        axes[1].plot(agg['J'], agg['bias_sq'], 'b--x', label='Bias$^2$')
        axes[1].plot(agg['J'], agg['var_err'], 'r--x', label='Variance')
        axes[1].axvline(Jc, color='gray', linestyle='dotted')
        axes[1].set_xlabel('Coupling $J$')
        axes[1].set_ylabel('Decoding Error')
        axes[1].set_title("Decoding Breakdown")
        axes[1].legend()
        

        axes[2].plot(agg['J'], agg['mean_rate'], 'g-o')
        axes[2].axvline(Jc, color='gray', linestyle='dotted')
        axes[2].set_xlabel('Coupling $J$')
        axes[2].set_ylabel('Mean Rate [Hz]')
        axes[2].set_title("Energetic Cost Proxy")
        
        plt.tight_layout()
        plt.savefig(REV_DIR / "figures" / f"FigR1_mechanism_fixed_N{N}.pdf")
        plt.close()

if __name__ == "__main__":
    run_diagnostics()
