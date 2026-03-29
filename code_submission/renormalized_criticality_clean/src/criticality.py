import numpy as np
from scipy.optimize import curve_fit
import warnings

def estimate_susceptibility(A_t: np.ndarray, dt: float, f0: float, eps: float, burn_in_s: float) -> float:
    
    if eps == 0.0:
        return 0.0
        
    N_steps = len(A_t)
    t = np.arange(N_steps) * dt
    

    burn_in_steps = int(burn_in_s / dt)
    if burn_in_steps >= N_steps:
        burn_in_steps = N_steps // 2
        
    mask = np.arange(burn_in_steps, N_steps)
    t_eff = t[mask]
    A_eff = A_t[mask] - np.mean(A_t[mask])
    
    T_eff = len(t_eff) * dt
    if T_eff <= 0:
        return 0.0
        
    response = np.sum(A_eff * np.exp(-1j * 2 * np.pi * f0 * t_eff)) * dt
    chi = (2.0 / (T_eff * eps)) * np.abs(response)
    
    return chi

def exponential_decay(t, a, tau):

    return a * np.exp(-t / tau)

def estimate_correlation_time(A_t: np.ndarray, dt: float, burn_in_s: float) -> tuple[float, float]:
   
    N_steps = len(A_t)
    burn_in_steps = int(burn_in_s / dt)
    if burn_in_steps >= N_steps:
        burn_in_steps = N_steps // 2
        
    A_eff = A_t[burn_in_steps:]
    A_center = A_eff - np.mean(A_eff)
    var = np.var(A_center)
    if var <= 0:
        return 0.0, 0.0
        
    n = len(A_center)
    f_A = np.fft.rfft(A_center, n=2*n)
    acf = np.fft.irfft(f_A * np.conjugate(f_A), n=2*n)[:n] / (var * n)
    
    max_lag_s = 2.0
    max_lag_steps = int(max_lag_s / dt)
    max_lag_steps = min(max_lag_steps, len(acf))
    
    acf_trunc = acf[:max_lag_steps]
    

    below_thresh = np.where(acf_trunc < 0.2)[0]
    cutoff = below_thresh[0] if len(below_thresh) > 0 else len(acf_trunc)
    

    tau_int = np.sum(acf_trunc[:cutoff]) * dt
    

    tau_fit = 0.0
    if cutoff > 2:
        t_lags = np.arange(cutoff) * dt
        y_lags = acf_trunc[:cutoff]
        try:
            p0 = (1.0, tau_int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(exponential_decay, t_lags, y_lags, p0=p0, bounds=([0, 0], [2.0, 10.0]))
                tau_fit = popt[1]
        except:
            tau_fit = tau_int
            
    return tau_fit, tau_int

def check_stability(
    A_t: np.ndarray, 
    dt: float,
    burn_in_s: float,
    r_max: float = 200.0, 
    var_max: float = 1e4, 
    dc_max_ratio: float = 0.99
) -> tuple[bool, dict]:
    
    N_steps = len(A_t)
    burn_in_steps = int(burn_in_s / dt)
    if burn_in_steps >= N_steps:
        burn_in_steps = N_steps // 2
    
    A_eff = A_t[burn_in_steps:]
    
    if np.any(np.isnan(A_eff)) or np.any(np.isinf(A_eff)):
        return False, {"error": "NaN or Inf", "mean_rate": np.nan, "var_rate": np.nan}
        
    mean_rate = np.mean(A_eff)
    var_rate = np.var(A_eff)
    
    metrics = {
        "mean_rate": mean_rate,
        "var_rate": var_rate
    }
    
    if mean_rate > r_max or mean_rate <= 0.0:
        return False, metrics
        
    if var_rate > var_max:
        return False, metrics
        

    A_freq = np.abs(np.fft.rfft(A_eff - np.mean(A_eff)))
    total_power = np.sum(A_freq**2)

    A_freq_raw = np.abs(np.fft.rfft(A_eff))
    total_power_raw = np.sum(A_freq_raw**2)
    
    if total_power_raw > 0 and (A_freq_raw[0]**2 / total_power_raw) > dc_max_ratio:
        return False, metrics
        
    return True, metrics

def select_critical_coupling(J_vals: np.ndarray, metric_vals: np.ndarray, stable_mask: np.ndarray) -> tuple[float, bool]:
    
    valid_indices = np.where(stable_mask)[0]
    if len(valid_indices) == 0:
        return np.nan, True
        
    last_stable_idx = valid_indices[-1]
    

    metric_valid = np.full_like(metric_vals, -np.inf)
    metric_valid[valid_indices] = metric_vals[valid_indices]
    
    peak_idx = np.argmax(metric_valid)
    if np.isinf(metric_valid[peak_idx]):
        return np.nan, True
        
    Jc = J_vals[peak_idx]
    edge_peak = False
    
    if peak_idx >= last_stable_idx - 1:

        edge_peak = True
        

    decreases_2 = True
    if peak_idx + 2 < len(metric_vals):
        if not (metric_vals[peak_idx + 1] < metric_vals[peak_idx] and metric_vals[peak_idx + 2] < metric_vals[peak_idx + 1]):
             decreases_2 = False
    else:
        decreases_2 = False
        
    if not decreases_2:
        edge_peak = True
        
    return Jc, edge_peak
