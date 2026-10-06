import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq

def generate_ou_stimulus(dt: float, T: float, tau_c: float, sigma_u: float, mu_u: float, seed: int) -> np.ndarray:
    """
    Generates an Ornstein-Uhlenbeck (OU) stimulus.
    
    dx = -(x - mu_u)/tau_c * dt + sigma_u * sqrt(2/tau_c) * dW
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T / dt))
    
    u = np.zeros(n_steps)
    u[0] = mu_u
    
    noise_amp = sigma_u * np.sqrt(2.0 / tau_c) * np.sqrt(dt)
    drift_coeff = dt / tau_c
    
    # Generate all noise at once for efficiency
    noise = rng.standard_normal(n_steps) * noise_amp
    
    for i in range(1, n_steps):
        u[i] = u[i-1] - (u[i-1] - mu_u) * drift_coeff + noise[i]
        
    return u

def generate_bandlimited_gaussian(dt: float, T: float, tau_c: float, var_req: float, seed: int) -> np.ndarray:
    """
    Generates band-limited Gaussian noise.
    White noise convolved with a Gaussian kernel of width ~ tau_c / dt.
    Normalized to desired variance `var_req`.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(np.ceil(T / dt))
    
    white_noise = rng.standard_normal(n_steps)
    
    # Gaussian kernel width
    sigma = tau_c / dt
    
    # Make a window large enough (e.g. 6 sigmas)
    window_size = int(6 * sigma)
    # Ensure odd window size for symmetry
    if window_size % 2 == 0:
        window_size += 1
        
    # Minimum window size
    if window_size < 3:
        window_size = 3
        
    window = signal.windows.gaussian(window_size, std=sigma)
    window /= np.sum(window)
    
    # Valid or same convolution? "same" keeps length equal
    u = signal.convolve(white_noise, window, mode='same')
    
    # Normalize to requested variance
    current_std = np.std(u)
    if current_std > 0:
        u = u * (np.sqrt(var_req) / current_std)
        
    return u

def effective_sampling_dt(u: np.ndarray, dt: float) -> tuple[float, float]:
    """
    Estimates effective Nyquist scale from PSD cutoff (heuristic).
    Returns (fc, dt_eff).
    """
    # PSD using Welch's method
    nperseg = min(len(u), int(2.0 / dt)) # 2 seconds window
    f, Pxx = signal.welch(u, fs=1.0/dt, nperseg=nperseg)
    
    # Heuristic: Find frequency where power drops to 5% of peak
    peak_power = np.max(Pxx)
    threshold = 0.05 * peak_power
    
    cutoff_indices = np.where(Pxx < threshold)[0]
    if len(cutoff_indices) > 0:
        # First frequency dipping below threshold after the peak
        peak_idx = np.argmax(Pxx)
        valid_cutoffs = cutoff_indices[cutoff_indices > peak_idx]
        if len(valid_cutoffs) > 0:
            fc = f[valid_cutoffs[0]]
        else:
            fc = f[-1] # fallback to Nyquist
    else:
        fc = f[-1]
        
    if fc > 0:
        dt_eff = 1.0 / (2.0 * fc)
    else:
        dt_eff = dt
        
    return fc, dt_eff

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    dt = 0.001
    T = 10.0
    tau_c = 0.05
    seed = 42
    
    u_ou = generate_ou_stimulus(dt, T, tau_c, sigma_u=1.0, mu_u=0.0, seed=seed)
    fc_ou, dt_eff_ou = effective_sampling_dt(u_ou, dt)
    
    u_bl = generate_bandlimited_gaussian(dt, T, tau_c, var_req=1.0, seed=seed)
    fc_bl, dt_eff_bl = effective_sampling_dt(u_bl, dt)
    
    print(f"OU: estimated fc = {fc_ou:.1f} Hz, dt_eff = {dt_eff_ou*1000:.1f} ms")
    print(f"BL: estimated fc = {fc_bl:.1f} Hz, dt_eff = {dt_eff_bl*1000:.1f} ms")
    
    # Plotting
    res_dir = Path("results/demo_stimulus")
    res_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    t_ax = np.arange(len(u_ou)) * dt
    
    ax[0, 0].plot(t_ax, u_ou)
    ax[0, 0].set_title("OU Stimulus")
    ax[0, 0].set_xlim(0, 2)
    
    f_ou, Pxx_ou = signal.welch(u_ou, fs=1.0/dt, nperseg=2048)
    ax[0, 1].semilogy(f_ou, Pxx_ou)
    ax[0, 1].axvline(fc_ou, color='r', linestyle='--', label=f'fc={fc_ou:.1f}Hz')
    ax[0, 1].set_title("OU PSD")
    ax[0, 1].set_xlim(0, max(fc_ou*3, 20))
    ax[0, 1].legend()
    
    ax[1, 0].plot(t_ax, u_bl)
    ax[1, 0].set_title("Band-limited Gaussian")
    ax[1, 0].set_xlim(0, 2)
    
    f_bl, Pxx_bl = signal.welch(u_bl, fs=1.0/dt, nperseg=2048)
    ax[1, 1].semilogy(f_bl, Pxx_bl)
    ax[1, 1].axvline(fc_bl, color='r', linestyle='--', label=f'fc={fc_bl:.1f}Hz')
    ax[1, 1].set_title("Band-limited PSD")
    ax[1, 1].set_xlim(0, max(fc_bl*3, 20))
    ax[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(res_dir / "stimulus_psd_demo.png")
    plt.close()
