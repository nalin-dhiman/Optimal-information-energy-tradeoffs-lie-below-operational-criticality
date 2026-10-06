"""Archival update plus independently injected voltage/log-hazard probes and observation-only summaries. Zero probes must reproduce the untouched source exactly."""
import numpy as np
import numba as nb
from typing import Optional, Dict

@nb.njit(fastmath=True)
def _simulate_two_path(
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
    seed: int,
    probe_voltage: np.ndarray,
    probe_hazard: np.ndarray
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
    W = np.zeros(N)
    meanV = np.zeros(n_steps)
    eligible = np.zeros(n_steps)
    features = np.zeros((n_steps, 3))
    factor_issue = np.zeros(n_steps, dtype=np.int8)
    
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
        base_argument = theta0 + thetaA*x
        if probe_hazard[t] != 0.0:
            base_argument += probe_hazard[t]
        base_rate = lambda0 * np.exp(base_argument)
        if probe_hazard[t] == 0.0:
            base_rate = lambda0 * np.exp(theta0 + thetaA * x)
        # Observation-only frozen-reset-path sensitivity to J. No extra RNG.
        count = 0; vv = 0.0; ww = 0.0
        for k in range(N):
            if r[k] >= tau_ref:
                count += 1; vv += V[k]; ww += W[k]
        features[t,0] = vv/count if count > 0 else 0.0
        features[t,1] = x
        features[t,2] = ww/count if count > 0 else 0.0
        
        for i in range(N):
            if r[i] < tau_ref:
                # Refractory
                r[i] += dt
                # V_i can optionally be clamped or evolve. Let's evolve it but block spikes.
            else:
                # Active
                rate_i = base_rate * np.exp(thetaV * V[i])
                if base_argument > 705.177555903787 or thetaV*V[i] > 709.782712893384 or base_argument+thetaV*V[i] > 705.177555903787:
                    factor_issue[t] = 1
                
                # Thinning (stable probability)
                p_spike = 1.0 - np.exp(-rate_i * dt)
                
                if np.random.random() < p_spike:
                    # Spike!
                    n_spikes_t += 1
                    r[i] = 0.0
                    if do_v_reset:
                        V[i] = v_reset_val
                        W[i] = 0.0
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
        if probe_voltage[t] != 0.0:
            target_V += probe_voltage[t]
        
        for i in range(N):
            noise_i = np.random.randn() * noise_amp
            V[i] = V[i] - (V[i] - target_V) * decay_v + noise_i
            W[i] = W[i] - W[i]*decay_v + x*decay_v
        vs = 0.0; es = 0
        for k in range(N):
            vs += V[k]
            if r[k] >= tau_ref: es += 1
        meanV[t] = vs/N
        eligible[t] = es/N

    mean_rate = mean_rate / (N * n_steps * dt)
    return A_t, x_t, mean_rate, meanV, eligible, features, factor_issue

