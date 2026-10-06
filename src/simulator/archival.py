import numpy as np
import numba as nb
from typing import Optional, Dict

@nb.njit(fastmath=True)
def _simulate_mc_numba(
    N: int, 
    n_steps: int, 
    dt: float, 
    u: np.ndarray, 
    J: float, 
    lambda0: float, 
    theta0: float, 
    thetaV: float, 
    thetaA: float, 
    tau_ref: float, 
    tau_s: float, 
    tau_v: float,
    v_reset_val: float,
    do_v_reset: bool,
    noise_sigma: float,
    seed: int
):
    """
    Numba-accelerated core of the Monte Carlo simulation.
    V_i evolves as OU process driven by u(t) + J * x(t) + internal noise.
    """
    np.random.seed(seed)
    
    # States
    V = np.zeros(N)
    r = np.zeros(N) # refractory age
    
    # Outputs
    A_t = np.zeros(n_steps)
    x_t = np.zeros(n_steps)
    mean_rate = 0.0
    
    x = 0.0
    
    # Precompute constants
    sqrt_dt = np.sqrt(dt)
    noise_amp = noise_sigma * np.sqrt(2.0 / tau_v) * sqrt_dt
    decay_v = dt / tau_v
    decay_s = dt / tau_s
    
    # Main temporal loop
    for t in range(n_steps):
        
        # 1. Spiking and refractory update
        n_spikes_t = 0
        
        # Calculate instantaneous rate (hazard)
        # lambda_i(t) = lambda0 * exp(theta0 + thetaV * V_i + thetaA * x(t))
        base_rate = lambda0 * np.exp(theta0 + thetaA * x)
        
        for i in range(N):
            if r[i] < tau_ref:
                # Refractory
                r[i] += dt
                # V_i can optionally be clamped or evolve. Let's evolve it but block spikes.
            else:
                # Active
                rate_i = base_rate * np.exp(thetaV * V[i])
                
                # Thinning (stable probability)
                p_spike = 1.0 - np.exp(-rate_i * dt)
                
                if np.random.random() < p_spike:
                    # Spike!
                    n_spikes_t += 1
                    r[i] = 0.0
                    if do_v_reset:
                        V[i] = v_reset_val
                else:
                    r[i] += dt
                    
        # 2. Update population variables
        # A(t) = spikes per dt per neuron
        A = (n_spikes_t / N) / dt
        A_t[t] = A
        mean_rate += n_spikes_t
        
        # x(t) follows tau_s dx/dt = -x + A(t)
        x = x - x * decay_s + A * dt / tau_s
        x_t[t] = x
        
        # 3. Update V_i
        # dV_i = -(V_i - (u(t) + J*x(t)))/tau_v * dt + noise
        target_V = u[t] + J * x
        
        for i in range(N):
            noise_i = np.random.randn() * noise_amp
            V[i] = V[i] - (V[i] - target_V) * decay_v + noise_i

    mean_rate = mean_rate / (N * n_steps * dt)
    return A_t, x_t, mean_rate


def run_simulation(
    N: int,
    dt: float,
    T: float,
    u: np.ndarray,
    J: float,
    lambda0: float = 1.0,
    theta0: float = -2.0,
    thetaV: float = 1.0,
    thetaA: float = 0.0,
    tau_ref: float = 0.005,
    tau_s: float = 0.02,
    tau_v: float = 0.01,
    v_reset_val: float = 0.0,
    do_v_reset: bool = True,
    noise_sigma: float = 0.5,
    seed: int = 42
) -> Dict:
    """
    Runs the Monte Carlo simulation of interacting renewal population.
    
    Returns a dictionary of results.
    """
    n_steps = int(np.ceil(T / dt))
    
    if len(u) < n_steps:
        raise ValueError(f"Stimulus u must have at least {n_steps} elements, but has {len(u)}")
        
    u = u[:n_steps]
        
    A_t, x_t, mean_rate = _simulate_mc_numba(
        N=N,
        n_steps=n_steps,
        dt=dt,
        u=u,
        J=J,
        lambda0=lambda0,
        theta0=theta0,
        thetaV=thetaV,
        thetaA=thetaA,
        tau_ref=tau_ref,
        tau_s=tau_s,
        tau_v=tau_v,
        v_reset_val=v_reset_val,
        do_v_reset=do_v_reset,
        noise_sigma=noise_sigma,
        seed=seed
    )
    
    # Compute summary statistics
    var_A = np.var(A_t)
    
    # Auto-correlation time (simple estimate from integrated auto-corr of A_t)
    A_center = A_t - np.mean(A_t)
    var_center = np.var(A_center)
    if var_center > 0:
        # Compute autocorrelation using FFT
        n = len(A_center)
        f_A = np.fft.fft(A_center, n=2*n)
        acf = np.fft.ifft(f_A * np.conjugate(f_A))[:n].real / (var_center * n)
        # integrate until first negative crossing or end
        neg_crossings = np.where(acf < 0)[0]
        cutoff = neg_crossings[0] if len(neg_crossings) > 0 else n
        tau_corr = np.sum(acf[:cutoff]) * dt
    else:
        tau_corr = 0.0
    
    results = {
        'A_t': A_t,
        'x_t': x_t,
        'u_t': u,
        'mean_rate': mean_rate,
        'var_A': var_A,
        'tau_corr': tau_corr,
        'dt': dt,
        'N': N
    }
    
    return results

if __name__ == '__main__':
    from src.stimulus import generate_ou_stimulus
    import time
    
    print("Testing Monte Carlo simulator...")
    dt = 0.0001 # 0.1 ms
    T = 10.0 # 10 seconds
    N = 10000 # 10k neurons
    
    print(f"Generating stimulus for T={T}s, dt={dt}s...")
    u = generate_ou_stimulus(dt=dt, T=T, tau_c=0.05, sigma_u=1.0, mu_u=0.0, seed=1)
    
    print(f"Running simulation with N={N}...")
    t0 = time.time()
    res = run_simulation(N=N, dt=dt, T=T, u=u, J=1.5, lambda0=100.0, seed=42)
    t1 = time.time()
    
    print(f"Simulation took {t1-t0:.2f} seconds.")
    print(f"Mean rate: {res['mean_rate']:.2f} Hz")
    print(f"Var(A): {res['var_A']:.4f}")
    print(f"Tau_corr: {res['tau_corr']*1000:.2f} ms")
