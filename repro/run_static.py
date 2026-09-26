"""Generate every stationary/spectral data file used by the revised manuscript."""
from pathlib import Path
import csv,json,platform,time
import numpy as np
import scipy
import pandas
import matplotlib
from scipy.linalg import eigh
from scipy.optimize import root_scalar,root
from model import *

BASE=Path(__file__).resolve().parents[1];DATA=BASE/'data';DATA.mkdir(exist_ok=True)
def dump(name,rows):
    with (DATA/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsdump(name,d):
    (DATA/name).write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))

def main():
    ranges={.6:range(24,32),.75:range(17,25),1.:range(10,18),1.25:range(8,16),1.5:range(6,14)}
    folds=[];theory=[]
    for beta,ms in ranges.items():
        for sigma in [1,-1]:
            x,C,K=fold_theory(beta,sigma);theory.append(dict(beta=beta,sigma=sigma,xstar=x,C=C,K=K))
            for m in ms:
                lat=Lattice(beta,m,sigma);q,v,row=lat.fold();folds.append(row)
                np.savez_compressed(DATA/f'fold_b{beta:g}_s{sigma}_m{m}.npz',n=lat.n,q=q,v=v,**row)
            print('folds',beta,sigma,'complete',flush=True)
    dump('folds_80.csv',folds);dump('fredholm_prefactors.csv',theory)
    conv=[]
    for beta,m,sigma in [(.6,31,1),(1,10,-1),(1.5,13,-1)]:
        for padding,tol in [(24,1e-9),(32,1e-11),(40,1e-11)]:
            lat=Lattice(beta,m,sigma,padding=padding);q,v,row=lat.fold(tol=tol)
            row.update(padding=padding,solver_xtol=tol);conv.append(row)
    dump('fold_convergence.csv',conv)
    spectral=[]
    for beta,ms in ranges.items():
        last=max(ms)
        for eta in [1.25,2.,4.]:
            for m in range(last-4,last+1):
                lat=Lattice(beta,m,-1);eps=eta*fold_theory(beta,-1)[1]*np.exp(-beta*m)
                q,res=lat.solve(eps);z=identify_modes(lat,q,eps);pred,x=frequency_theory(beta,m,eta)
                row=dict(beta=beta,eta=eta,m=m,epsilon=eps,Rfit=z['Rfit'],x_asym=x,residual=res,points=len(q))
                for j,name in enumerate(['exchange','pinning','stretch']):
                    row[name]=z['frequencies'][j];row[name+'_theory']=pred[j];row[name+'_error']=abs(pred[j]/z['frequencies'][j]-1)
                    row[name+'_overlap']=z['overlaps'][j,j];row[name+'_krein']=z['krein'][j]
                spectral.append(row)
            print('spectral',beta,eta,'complete',flush=True)
    dump('spectral_grid.csv',spectral)
    kappas=[]
    for b in ranges:
        for eta in [1.25,2.,4.]:
            rr=[r for r in spectral if r['beta']==b and r['eta']==eta]
            for a,brow in zip(rr[:-1],rr[1:]):
                r=dict(beta=b,eta=eta,m=a['m'],m_next=brow['m'])
                for name in ['exchange','pinning','stretch']:r[name]=-2*np.log(brow[name]/a[name])
                kappas.append(r)
    dump('spectral_exponents.csv',kappas)

    reps={};sc=[]
    for padding in [24,32,40]:
        lat=Lattice(1,12,-1,padding=padding);eps=2*fold_theory(1,-1)[1]*np.exp(-12);q,res=lat.solve(eps);z=identify_modes(lat,q,eps)
        sc.append(dict(padding=padding,points=len(q),residual=res,exchange=z['frequencies'][0],pinning=z['frequencies'][1],stretch=z['frequencies'][2],eig_residual=max(z['eig_residuals'])))
        if padding==32:
            np.savez_compressed(DATA/'representative_stable.npz',n=lat.n,q=q,epsilon=eps,**z)
            reps.update(epsilon_stable=eps,frequencies=z['frequencies'],overlaps=z['overlaps'],Rfit=z['Rfit'],Krein=z['krein'],Rmol=geometry(lat.n,q)['Rmol'])
    dump('spectral_convergence.csv',sc)

    # Symmetric fold branches and their complete unconstrained spectra.
    lat=Lattice(1,10,-1);qf,vf,fold=lat.fold();ef=fold['epsilon'];rows=[]
    epsvals=ef+np.r_[np.geomspace(1e-7,1e-3,18),np.linspace(.0012,.05-ef,45)]
    starts=[]
    for sign in [-1,1]:
        amp=np.sqrt(-2*fold['transversality']*(epsvals[0]-ef)/fold['quadratic'])
        q,_=lat.solve(epsvals[0],q0=qf+sign*amp*vf);starts.append(q)
    starts=sorted(starts,key=lambda q:geometry(lat.n,q)['Rmol'])
    branchprofiles={}
    for name,q0 in zip(['main','companion'],starts):
        for e in epsvals:
            q,res=lat.solve(e,q0=q0);q0=q
            ev,vr,A=spectrum(q,1,e);g=geometry(lat.n,q)
            rows.append(dict(branch=name,epsilon=e,Rmol=g['Rmol'],max_real=max(ev.real),residual=res))
        branchprofiles[name]=q
    dump('branches_m10.csv',rows)
    # Secondary zero in the opposite (even) reflection sector.
    pe=Lattice(1,10,1).P
    def sbfun(e,ret=False):
        q,res=lat.solve(e,x=.19);d=np.sqrt(1+q*q);S=-stationary_J(q,1,e)/d[:,None]*d[None,:]
        vals,vec=eigh(pe.T@S@pe);k=np.argmin(abs(vals))
        return (q,pe@vec[:,k]*d,vals[k]) if ret else vals[k]
    esb=root_scalar(sbfun,bracket=(.0095,.0101),xtol=3e-14).root
    qsb,vsb,zz=sbfun(esb,True);vsb/=np.linalg.norm(vsb)
    reps.update(epsilon_fold=ef,epsilon_SB=esb,SB_even_parity_residual=np.linalg.norm(vsb-vsb[::-1]))
    np.savez_compressed(DATA/'symmetry_breaking.npz',n=lat.n,q=qsb,v=vsb,epsilon=esb)
    def center(q):
        rho=np.log1p(q*q);N=sum(rho);return np.dot(lat.n,rho)/N
    def centerd(q):
        rho=np.log1p(q*q);N=sum(rho);c=np.dot(lat.n,rho)/N;return 2*q/(1+q*q)*(lat.n-c)/N
    daughter=[];previous=None
    for target in np.r_[.002,.005,.01,np.linspace(.02,.20,13)]:
        if previous is None:z0=np.r_[qsb+vsb*target/np.dot(centerd(qsb),vsb),esb+target**2*.02]
        else:z0=previous.copy();z0[:-1]+=vsb*(target-center(z0[:-1]))/np.dot(centerd(z0[:-1]),vsb)
        def f(z):return np.r_[stationary_F(z[:-1],1,z[-1]),center(z[:-1])-target]
        def j(z):
            q=z[:-1];out=np.zeros((len(q)+1,len(q)+1));out[:-1,:-1]=stationary_J(q,1,z[-1]);out[:-1,-1]=2*q**3;out[-1,:-1]=centerd(q);return out
        sol=root(f,z0,jac=j,tol=1e-11)
        if max(abs(f(sol.x)))>1e-10:raise RuntimeError(('daughter',target,sol.message))
        q=sol.x[:-1];e=sol.x[-1];previous=sol.x;ev,vr,A=spectrum(q,1,e)
        g=geometry(lat.n,q)
        daughter.append(dict(A=target,epsilon=e,Rmol=g['Rmol'],norm_asymmetry=g['asym'],max_real=max(ev.real),residual=max(abs(stationary_F(q,1,e)))))
    dump('daughter_branches.csv',daughter)

    # Finite-epsilon Hamiltonian-Hopf continuation, on m=5 branch.
    hh=Lattice(1,5,-1)
    eps0=hh_threshold();hhrows=[]
    for e in eps0+np.r_[np.linspace(-3e-4,-1e-6,25),np.linspace(1e-6,1.2e-4,25)]:
        q,res=hh.solve(e,x=.1);ev,vr,A=spectrum(q,1,e)
        ids=np.where((ev.imag>.39)&(ev.imag<.44))[0]
        ids=ids[np.argsort(ev[ids].real if e>eps0 else ev[ids].imag)]
        if len(ids)!=2:raise RuntimeError(('HH modes',e,ev[ids]))
        row=dict(epsilon=e,residual=res)
        for k,jj in enumerate(ids):
            row[f'real_{k}']=ev[jj].real;row[f'imag_{k}']=ev[jj].imag;row[f'krein_{k}']=krein(q,1,e,vr[:,jj])
        hhrows.append(row)
    dump('hamiltonian_hopf.csv',hhrows)
    e=eps0+1e-4;q,res=hh.solve(e,x=.1);ev,vr,A=spectrum(q,1,e);k=np.argmax(np.where(ev.imag>0,ev.real,-999))
    np.savez_compressed(DATA/'representative_HH.npz',n=hh.n,q=q,epsilon=e,ev=ev,vr=vr)
    reps.update(epsilon_HH_refined=eps0,HH_check_epsilon=e,HH_real=ev[k].real,HH_imag=ev[k].imag)
    lat=Lattice(1,10,-1);q,res=lat.solve(.0095,x=.20);ev,vr,A=spectrum(q,1,.0095)
    np.savez_compressed(DATA/'representative_real_unstable.npz',n=lat.n,q=q,epsilon=.0095,ev=ev,vr=vr)
    reps['real_growth']=max(ev.real)
    jsdump('representative_values.json',reps)
    jsdump('environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,pandas=pandas.__version__,matplotlib=matplotlib.__version__,precision='float64 / complex128',stationary_padding=32,stationary_xtol=1e-11,fold_xtol=1e-10))
    print('STATIC COMPLETE',json.dumps(reps,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)),flush=True)

if __name__=='__main__':main()
