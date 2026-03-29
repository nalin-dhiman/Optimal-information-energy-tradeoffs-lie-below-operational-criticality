import numpy as np
from typing import Tuple, Optional

def jensen_hazard_mean(mu: float, var: float, params: dict = None) -> float:
    
    return np.exp(mu + 0.5 * var)


def predict_Jc_jensen(
    tau_s: float, 
    tau_v: float, 
    thetaV: float, 
    thetaA: float, 
    mu_N: float, 
    var_N: float
) -> float:
    
    return 1.0 / (thetaV * np.sqrt(var_N) + 1e-6)

def fit_scaling(N_vals: np.ndarray, Jc_vals: np.ndarray) -> Tuple[float, float]:
   
    valid_idx = ~np.isnan(Jc_vals) & (Jc_vals > 0)
    if np.sum(valid_idx) < 2:
        return 0.0, 0.0
        
    x = np.log(N_vals[valid_idx])
    y = np.log(Jc_vals[valid_idx])
    
    coeffs = np.polyfit(x, y, 1) 
    
    exponent = coeffs[0]
    intercept = coeffs[1]
    
    return exponent, intercept
