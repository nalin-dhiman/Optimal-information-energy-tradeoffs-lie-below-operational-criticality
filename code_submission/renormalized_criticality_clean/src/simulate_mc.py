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
   
    np.random.seed(seed)
    

    V = np.zeros(N)
    r = np.zeros(N)
    

    A_t = np.zeros(n_steps)
    x_t = np.zeros(n_steps)
    mean_rate = 0.0
    
    x = 0.0
    

    sqrt_dt = np.sqrt(dt)
    noise_amp = noise_sigma * np.sqrt(2.0 / tau_v) * sqrt_dt
    decay_v = dt / tau_v
    decay_s = dt / tau_s
    

    for t in range(n_steps):
        

        n_spikes_t = 0
        

        base_rate = lambda0 * np.exp(theta0 + thetaA * x)
        
        for i in range(N):
            if r[i] < tau_ref:

                r[i] += dt

            else:

                rate_i = base_rate * np.exp(thetaV * V[i])
                

                p_spike = 1.0 - np.exp(-rate_i * dt)
                
                if np.random.random() < p_spike:

                    n_spikes_t += 1
                    r[i] = 0.0
                    if do_v_reset:
                        V[i] = v_reset_val
                else:
                    r[i] += dt

        A = (n_spikes_t / N) / dt
        A_t[t] = A
        mean_rate += n_spikes_t
        

        x = x - x * decay_s + A * dt / tau_s
        x_t[t] = x
        
       
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
    

    var_A = np.var(A_t)
    

    A_center = A_t - np.mean(A_t)
    var_center = np.var(A_center)
    if var_center > 0:

        n = len(A_center)
        f_A = np.fft.fft(A_center, n=2*n)
        acf = np.fft.ifft(f_A * np.conjugate(f_A))[:n].real / (var_center * n)

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
    dt = 0.0001 
    T = 10.0
    N = 10000 
    
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
