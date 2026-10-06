import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from src.simulator.archival import _simulate_mc_numba
from src.simulator.observed import _simulate_observed
from src.simulator.diagnostic import _simulate_two_path
from src.analysis.reader import causal_features,integer_bins,fit_ridge,predict,score
from src.analysis.evaluation import stage3_seed,trajectory,evaluate_fixed
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('J,theta',[(0,[0,1,0]),(.26,[3.825831,3.952520,-.491078]),(.32,[1.819837,4.034597,-.631138])])
def test_instrumentation_matches_archival(J,theta):
    u=np.linspace(-.2,.2,200);args=(80,200,.0005,u,J,100.,*theta,.005,.02,.01,0.,True,.5,21)
    old=_simulate_mc_numba(*args)
    for q in [_simulate_observed(*args),_simulate_two_path(*args,np.zeros(200),np.zeros(200))]:
        assert np.array_equal(old[0],q[0]) and np.array_equal(old[1],q[1]) and old[2]==q[2]

def test_source_hashes():
    for p,h in json.loads((ROOT/'provenance/archival_source_hashes.json').read_text()).items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h

def test_voltage_probe_matches_drive():
    u=np.linspace(-.2,.2,200);v=.01*np.cos(np.arange(200)*.1);z=np.zeros(200)
    args=(80,200,.0005,u,.26,100.,3.825831,3.952520,-.491078,.005,.02,.01,0.,True,.5,21)
    q=_simulate_two_path(*args,v,z);args=list(args);args[3]=u+v;r=_simulate_mc_numba(*args)
    assert np.array_equal(q[0],r[0]) and np.array_equal(q[1],r[1])

def test_hazard_probe_matches_intercept():
    u=np.zeros(200);args=[80,200,.0005,u,.26,100.,3.825831,3.952520,-.491078,.005,.02,.01,0.,True,.5,21]
    q=_simulate_two_path(*args,u,np.full(200,.01));args[6]+=.01;r=_simulate_mc_numba(*args)
    assert np.array_equal(q[0],r[0]) and np.array_equal(q[1],r[1])

def test_rate_and_refractory_units():
    u=np.zeros(200);args=(80,200,.0005,u,0,100.,30.,0.,0.,.005,.02,.01,0.,True,0.,21)
    A,x,r,*_=_simulate_observed(*args);i=np.flatnonzero(A)
    assert np.all(np.diff(i)*.0005>=.0055-1e-12) and np.all(A[i]==2000.)
    assert r==pytest.approx(A.mean())

def test_causal_reader_clock_and_future_latency():
    u=np.arange(1000.);x=2*u
    X,y,c=causal_features(u,x,dt=.0005,row_s=.005,burn_s=.05,history_s=.02,fir_taps=1,cutoff_Hz=80)
    assert np.array_equal(X[:,0],2*y) and np.array_equal(X[:,-1],2*(y-40))
    _,future,d=causal_features(u,x,dt=.0005,row_s=.005,burn_s=.05,history_s=.02,latency_s=.005,fir_taps=1,cutoff_Hz=80)
    assert np.array_equal(d['target_indices'],d['feature_indices']+10)
    x2=x.copy();x2[800:]=9999
    X2,_,_=causal_features(u,x2,dt=.0005,row_s=.005,burn_s=.05,history_s=.02)
    X1,_,_=causal_features(u,x,dt=.0005,row_s=.005,burn_s=.05,history_s=.02)
    assert np.array_equal(X1[c['feature_indices']<800],X2[c['feature_indices']<800])

@pytest.mark.parametrize('duration,dt',[(.0033,.0005),(.011,.005),(-.01,.005)])
def test_nonphysical_clocks_rejected(duration,dt):
    with pytest.raises(ValueError):integer_bins(duration,dt)

def test_train_only_standardization_and_risk():
    X=np.arange(80.).reshape(40,2);y=3*X[:,0]+2
    m=fit_ridge(X,y,.0001);stored=m['mean'].copy();p=predict(m,X+10)
    assert np.array_equal(m['mean'],stored)
    s=score(y,p);assert s['R2']==pytest.approx(1-s['normalized_MSE'])

def test_seed_splits_and_true_no_recurrence():
    seeds=[stage3_seed('stage3_input',s,0,40.) for s in ['selection_train','selection_validation','outer']]
    assert len(set(seeds))==3
    c=json.loads((ROOT/'configs/smoke.json').read_text())
    with pytest.raises(ValueError,match='theta_A'):trajectory(c,dict(J=0.,theta=[0,1,-.1],true_no_recurrence=True),'outer',0,2.)

def test_fixed_network_evaluation_keeps_independent_splits():
    c=json.loads((ROOT/'configs/smoke.json').read_text());row,raw,candidates=evaluate_fixed(c,c['networks'][0],0)
    assert row['split']=='outer' and row['block']==0 and np.isfinite(row['normalized_MSE'])
    assert not np.array_equal(raw['train']['u'],raw['outer']['u'])
    assert len(candidates)==len(c['ridge_alpha'])
