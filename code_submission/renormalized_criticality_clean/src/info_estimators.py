import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from typing import Dict, Tuple

def build_lagged_features(A_t: np.ndarray, lags: int, step: int) -> np.ndarray:
   
    N = len(A_t)
    eff_len = N - lags * step
    if eff_len <= 0:
        raise ValueError("Time series too short for requested lags.")
        
    X = np.zeros((eff_len, lags))
    for i in range(lags):

        idx_start = i * step
        idx_end = idx_start + eff_len
        X[:, lags - 1 - i] = A_t[idx_start:idx_end]
        
    return X


def estimate_information_decoder(
    A_t: np.ndarray, 
    u_t: np.ndarray, 
    dt_eff: float,
    lags: int = 20, 
    lag_step: int = 1,
    n_splits: int = 5,
    alpha: float = 1.0,
    shuffle_null: bool = False
) -> Tuple[float, float, Dict]:
    
    try:
        X = build_lagged_features(A_t, lags=lags, step=lag_step)
    except ValueError:
        return 0.0, 0.0, {"error": "Too few samples"}
        

    offset = lags * lag_step
    y = u_t[offset:]
    
    if len(y) != X.shape[0]:
        min_len = min(len(y), X.shape[0])
        X = X[:min_len]
        y = y[:min_len]
        
    if shuffle_null:

        block_size = max(1, int(0.5 / dt_eff)) 
        n_blocks = len(y) // block_size
        indices = np.arange(n_blocks * block_size)
        blocks = indices.reshape(n_blocks, block_size)
        np.random.shuffle(blocks)
        shuffled_indices = blocks.flatten()
        

        rem = len(y) - len(shuffled_indices)
        if rem > 0:
            shuffled_indices = np.concatenate([shuffled_indices, np.arange(len(shuffled_indices), len(y))])
            
        y = y[shuffled_indices]
        

    kf = KFold(n_splits=n_splits, shuffle=False)
    
    mses = []
    vars_u = []
    

    gap = min(int(0.1 / dt_eff), len(y)//10) 
    
    for train_index, test_index in kf.split(X):

        test_start = test_index[0]
        test_end = test_index[-1]
        
        valid_mask = (train_index < test_start - gap) | (train_index > test_end + gap)
        valid_train = train_index[valid_mask]
        
        if len(valid_train) == 0:
            continue
            
        X_train, y_train = X[valid_train], y[valid_train]
        X_test, y_test = X[test_index], y[test_index]
        
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = np.mean((y_test - y_pred)**2)
        var_y = np.var(y_test)
        
        mses.append(mse)
        vars_u.append(var_y)
        
    if len(mses) == 0:
        return 0.0, 0.0, {"error": "CV failed"}
        
    mean_mse = np.mean(mses)
    mean_var = np.mean(vars_u)
    
   
    if mean_mse > 0 and mean_var > mean_mse:
        bits_per_sample = 0.5 * np.log2(mean_var / mean_mse)
    else:
        bits_per_sample = 0.0
        
   
    i_dec = bits_per_sample * (1.0 / dt_eff)
    
    diagnostics = {
        "bits_per_sample": bits_per_sample,
        "n_samples": len(y),
        "gap": gap,
        "raw_mse": mean_mse,
        "raw_var": mean_var
    }
    
    return i_dec, mean_mse, diagnostics

