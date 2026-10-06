"""Bit equality of observational/zero-probe instrumentation against archival source."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.simulator.archival import _simulate_mc_numba
from src.simulator.observed import _simulate_observed
from src.simulator.diagnostic import _simulate_two_path
def main():
    for J,theta in [(0.,[0.,1.,0.]),(.26,[3.825831,3.952520,-.491078]),(.32,[1.819837,4.034597,-.631138])]:
        u=np.linspace(-.2,.2,200);args=(80,200,.0005,u,J,100.,*theta,.005,.02,.01,0.,True,.5,2301)
        old=_simulate_mc_numba(*args);observed=_simulate_observed(*args);probe=_simulate_two_path(*args,np.zeros(200),np.zeros(200))
        for new in [observed,probe]:
            assert np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1]) and old[2]==new[2]
    print('3 configurations: archival/observed/zero-probe arrays bit-identical')
if __name__=='__main__':main()
