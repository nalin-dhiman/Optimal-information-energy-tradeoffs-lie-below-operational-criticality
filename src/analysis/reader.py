"""Causal FIR measurement and lag features on an explicit physical clock."""
import numpy as np
from scipy import signal, linalg

def integer_bins(seconds, dt):
    ratio=seconds/dt; bins=round(ratio)
    if bins<0 or not np.isclose(ratio,bins,rtol=0,atol=1e-9):
        raise ValueError('Duration must be an integer number of physical bins')
    return bins

def causal_features(u,x,dt=.0005,row_s=.005,burn_s=4.,history_s=.1,latency_s=0.,fir_taps=41,cutoff_Hz=80.):
    u=np.asarray(u);x=np.asarray(x)
    if u.ndim!=1 or x.shape!=u.shape:raise ValueError('Input and response must be matching 1D arrays')
    stride=integer_bins(row_s,dt)
    if stride<1:raise ValueError('Row spacing must be at least one simulator bin')
    h=integer_bins(history_s,row_s);lag=integer_bins(latency_s,dt);burn=integer_bins(burn_s,dt)
    fir=signal.firwin(fir_taps,cutoff_Hz,fs=1/dt);filtered=signal.lfilter(fir,[1.],x)
    rows=np.arange(burn,len(u)-lag,stride);valid=np.arange(h,len(rows))
    if not len(valid):raise ValueError('Trajectory too short for burn, history and latency')
    recent=rows[valid];X=np.column_stack([filtered[rows[valid-k]] for k in range(h+1)])
    target=recent+lag
    return X,u[target],dict(feature_indices=recent,target_indices=target,physical_row_s=row_s,history_s=history_s,additional_target_latency_s=latency_s,FIR_group_delay_s=(fir_taps-1)*dt/2)

def fit_ridge(X,y,alpha):
    if alpha<=0:raise ValueError('Ridge alpha must be positive')
    mu=X.mean(0);sd=X.std(0);sd[sd<1e-10]=1.;Z=(X-mu)/sd;ym=float(y.mean())
    w=linalg.solve(Z.T@Z/len(y)+alpha*np.eye(X.shape[1]),Z.T@(y-ym)/len(y),assume_a='pos')
    return dict(mean=mu,sd=sd,w=w,ym=ym,alpha=float(alpha))

def predict(model,X):return ((X-model['mean'])/model['sd'])@model['w']+model['ym']
def score(y,p):
    mse=float(np.mean((y-p)**2));v=float(np.var(y))
    if v<=0:raise ValueError('Normalized risk requires a nonconstant target')
    return dict(MSE=mse,normalized_MSE=mse/v,R2=1-mse/v,target_variance=v)
def select_reader(train,validation,alphas):
    X,y=train;V,z=validation;models=[fit_ridge(X,y,a) for a in alphas]
    outcomes=[score(z,predict(m,V)) for m in models];best=int(np.argmin([s['MSE'] for s in outcomes]))
    return models[best],outcomes
