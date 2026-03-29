import numpy as np
from typing import Dict, Tuple, List, Callable
from scipy.optimize import minimize
from  src.simulate_mc import run_simulation
from  src.info_estimators import estimate_information_decoder

def objective_function(
    params: np.ndarray,
    N: int,
    dt: float,
    T: float,
    u: np.ndarray,
    dt_eff: float,
    J: float,
    beta_E: float,
    beta_C: float,
    lambda0: float = 100.0,
    seed: int = 42
) -> Tuple[float, Dict]:
    
    theta0, thetaV, thetaA = params
    
    
    res = run_simulation(
        N=N, 
        dt=dt, 
        T=T, 
        u=u, 
        J=J, 
        lambda0=lambda0, 
        theta0=theta0, 
        thetaV=thetaV, 
        thetaA=thetaA, 
        seed=seed
    )
    

    A_t = res['A_t']
    mean_rate = res['mean_rate']
    

    if mean_rate < 0.1 or mean_rate > 500.0:
        return 1e6, {"error": "unstable"}
        
    i_dec, mse, _ = estimate_information_decoder(
        A_t=A_t,
        u_t=u,
        dt_eff=dt_eff,
        lags=20,
        lag_step=max(1, int(0.005 / dt)) 
    )
    
    l1_norm = np.sum(np.abs(params))
    
    L = i_dec - beta_E * mean_rate - beta_C * l1_norm

    
    diagnostics = {
        'i_dec': i_dec,
        'mean_rate': mean_rate,
        'l1_norm': l1_norm,
        'L': L
    }
    
    return -L, diagnostics

def objective_wrapper(params, *args):

    val, _ = objective_function(params, *args)
    return val

def optimize_theta(
    N: int,
    dt: float,
    T: float,
    u: np.ndarray,
    dt_eff: float,
    J: float,
    beta_E: float,
    beta_C: float,
    n_restarts: int = 3,
    seed: int = 42,
    maxiter: int = 50
) -> Dict:
    
    best_loss = np.inf
    best_params = None
    best_diagnostics = None
    
    np.random.seed(seed)
    

    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    

    initial_guesses = [
        np.array([-2.0, 1.0, 0.0]), 
        np.array([-3.0, 2.0, -1.0]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    args = (N, dt, T, u, dt_eff, J, beta_E, beta_C, 100.0, seed)
    
    for i in range(min(n_restarts, len(initial_guesses))):
        guess = initial_guesses[i]
        

        res = minimize(
            objective_wrapper,
            x0=guess,
            args=args,
            method='Powell',
            bounds=bounds,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        if res.success or res.fun < best_loss: 
            _, diag = objective_function(res.x, *args)
            if res.fun < best_loss and "error" not in diag:
                best_loss = res.fun
                best_params = res.x
                best_diagnostics = diag
                
    if best_params is None:
        best_params = initial_guesses[0]
        _, best_diagnostics = objective_function(best_params, *args)
        
    return {
        'best_params': best_params.tolist(),
        'objective': -best_loss,
        'diagnostics': best_diagnostics
    }

def optimize_over_J(
    J_vals: np.ndarray,
    N: int,
    dt: float,
    T: float,
    u: np.ndarray,
    dt_eff: float,
    beta_E: float,
    beta_C: float,
    n_restarts: int = 3,
    seed: int = 42,
    maxiter: int = 50
) -> Dict:
    
    results = []
    
    for J in J_vals:
        res = optimize_theta(
            N=N, dt=dt, T=T, u=u, dt_eff=dt_eff, 
            J=J, beta_E=beta_E, beta_C=beta_C, 
            n_restarts=n_restarts, seed=seed, maxiter=maxiter
        )
        res['J'] = float(J)
        results.append(res)
        

    best_idx = np.argmax([r['objective'] for r in results])
    J_star = float(J_vals[best_idx])
    
    return {
        'J_star': J_star,
        'results': results
    }
