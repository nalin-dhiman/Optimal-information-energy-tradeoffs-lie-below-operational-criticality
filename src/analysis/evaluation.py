"""Replay a fixed network; no optimizer or test-set parameter selection."""
import hashlib,json
import numpy as np
from src.simulator.observed import _simulate_observed
from src.simulator.stimulus import generate_ou_stimulus
from src.analysis.reader import causal_features,select_reader,predict,score

def stage3_seed(*args):
    return int(hashlib.sha256(json.dumps(args).encode()).hexdigest()[:8],16)%2147483646+1

def trajectory(config,network,split,block,T):
    dt=config['dt_s'];burn=config['burn_s'];total=float(T)+burn
    si=stage3_seed('stage3_input',split,block,float(T));sn=stage3_seed('stage3_intrinsic',split,block,float(T))
    u=generate_ou_stimulus(dt,total,config['input_tau_s'],config['input_sigma'],0.,si)
    theta=network['theta'];J=network['J']
    if network.get('true_no_recurrence') and (J!=0 or theta[2]!=0):raise ValueError('True no recurrence requires J=0 and theta_A=0')
    result=_simulate_observed(config['N'],len(u),dt,u,J,config['lambda0_Hz'],*theta,config['tau_ref_s'],config['tau_s_s'],config['tau_v_s'],config['v_reset'],config['reset'],config['noise_sigma'],sn)
    A,x,_,V,e,bad=result;post=round(burn/dt)
    if np.any(bad[post:]) or not all(np.isfinite(q).all() for q in [A,x,V,e]):raise FloatingPointError('Numerically ineligible fixed-network replay')
    X,y,clock=causal_features(u,x,dt,config['row_s'],burn,config['history_s'],config['latency_s'],config['fir_taps'],config['cutoff_Hz'])
    meta=dict(input_seed=si,neural_seed=sn,split=split,block=block,T_s=T,mean_rate_Hz=float(A[post:].mean()),source_response_sha256=hashlib.sha256(x.tobytes()).hexdigest(),**{k:v for k,v in clock.items() if not isinstance(v,np.ndarray)})
    return X,y,meta,dict(u=u,A=A,x=x,Vmean=V,eligible=e)

def evaluate_fixed(config,network,block):
    train=trajectory(config,network,'selection_train',block,config['train_s'])
    val=trajectory(config,network,'selection_validation',block,config['validation_s'])
    reader,candidates=select_reader(train[:2],val[:2],config['ridge_alpha'])
    outer=trajectory(config,network,'outer',block,config['outer_s'])
    row=dict(outer[2]);row.update(network=network['name'],J=network['J'],theta0=network['theta'][0],thetaV=network['theta'][1],thetaA=network['theta'][2],alpha=reader['alpha'],**score(outer[1],predict(reader,outer[0])))
    return row,dict(train=train[3],validation=val[3],outer=outer[3]),candidates
