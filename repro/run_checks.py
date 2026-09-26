"""Additional finite-domain HH refinement and collision sampling check."""
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from scipy.optimize import root_scalar
from model import *
from run_static import dump
from run_dynamics import collision

def main():
    rows=[]
    for padding in [24,32,40]:
        lat=Lattice(1,5,-1,padding=padding)
        def discr(e,ret=False):
            q,res=lat.solve(e,x=.1)
            mu=eigvals(-stationary_J(q,1,e,True)@stationary_J(q,1,e))
            ids=np.argsort(abs(mu+.4128**2))[:2];pair=mu[ids]
            value=float(np.real((pair[0]-pair[1])**2))
            return (value,pair,q,res) if ret else value
        e=root_scalar(discr,bracket=(.52670,.52673),xtol=1e-13).root
        value,pair,q,res=discr(e,True)
        ev,vr,A=spectrum(q,1,e)
        lo=e-1e-7;hi=e+1e-7
        rows.append(dict(padding=padding,points=len(q),epsilon_HH=e,frequency=np.sqrt(-pair.mean().real),
                         discriminant=value,bracket_below=lo,discriminant_below=discr(lo),
                         bracket_above=hi,discriminant_above=discr(hi),residual=res))
    dump('HH_convergence.csv',rows)
    row=collision((np.pi,.0062,200,2e-11,2e-13,5000,'dense_output',.5))
    extra=collision((0.,.0075,400,2e-11,2e-13,5000,'coarse_max_check',.5))
    dump('collision_output_check.csv',[row,extra])
    print('CHECKS COMPLETE',rows,flush=True)

if __name__=='__main__':main()
