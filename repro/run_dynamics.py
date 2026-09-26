"""Independent direct integration and explicitly resolved collision scan."""
from pathlib import Path
import csv,json,time,argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from model import *

BASE=Path(__file__).resolve().parents[1];DATA=BASE/'data';DATA.mkdir(exist_ok=True)
def dump(name,rows):
    with (DATA/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsdump(name,d):
    (DATA/name).write_text(json.dumps(d,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else float(x)))

def rhs(t,z,beta,eps):
    return 1j*(-2*np.cosh(beta)*z+(1+abs(z)**2)*neighbors(z)+2*eps*abs(z)**2*z)

def invariants_series(z,eps):
    a=abs(z)**2;norm=np.sum(np.log1p(a),axis=0)
    H=-2*np.real(np.sum(z[:-1]*np.conj(z[1:]),axis=0))-2*eps*np.sum(a-np.log1p(a),axis=0)
    return norm,H

def mode_dns(kind,rtol=1e-10,atol=1e-12,amplitude=1e-6):
    if kind.startswith('stable'):
        d=np.load(DATA/'representative_stable.npz');beta=1.;eps=float(d['epsilon']);q=d['q'];n=d['n'];j=int(kind[-1]);target=d['frequencies'][j]
        duration=6*2*np.pi/target
    else:
        d=np.load(DATA/('representative_HH.npz' if kind=='HH' else 'representative_real_unstable.npz'))
        beta=1.;eps=float(d['epsilon']);q=d['q'];n=d['n'];duration=600 if kind=='HH' else 200
    ev,vr,vl,A=spectrum(q,beta,eps,left=True)
    if kind.startswith('stable'):k=np.argmin(abs(ev-1j*target))
    elif kind=='HH':k=np.argmax(np.where(ev.imag>0,ev.real,-999))
    else:k=np.argmax(ev.real)
    w=vr[:,k];w=w*np.exp(-1j*np.angle(w[np.argmax(abs(w))]));left=vl[:,k];norm=np.vdot(left,w)
    physical=w.real
    z0=q+amplitude*(physical[:len(q)]+1j*physical[len(q):])
    ts=np.linspace(0,duration,2401)
    start=time.perf_counter()
    sol=solve_ivp(rhs,(0,duration),z0,args=(beta,eps),method='DOP853',rtol=rtol,atol=atol,t_eval=ts)
    if not sol.success:raise RuntimeError(sol.message)
    y=np.r_[sol.y.real-q[:,None],sol.y.imag]
    coeff=left.conj()@y/norm
    logamp=np.log(abs(coeff));angle=np.unwrap(np.angle(coeff))
    sl=slice(30,-30)
    growth=np.polyfit(ts[sl],logamp[sl],1)[0];frequency=np.polyfit(ts[sl],angle[sl],1)[0]
    N,H=invariants_series(sol.y,eps)
    rho=np.log1p(abs(sol.y)**2);le=n<0;ri=n>0
    nl=rho[le].sum(axis=0);nr=rho[ri].sum(axis=0)
    XL=(n[le,None]*rho[le]).sum(axis=0)/nl;XR=(n[ri,None]*rho[ri]).sum(axis=0)/nr
    center=(n[:,None]*rho).sum(axis=0)/N
    label=f'{kind}_a{amplitude:g}_tol{rtol:g}'
    np.savez_compressed(DATA/f'dns_{label}.npz',t=ts,c=coeff,N=N,H=H,Rmol=XR-XL,center=center,deltaN=nr-nl,lambda_bdg=ev[k],amplitude=amplitude)
    row=dict(kind=kind,amplitude=amplitude,rtol=rtol,atol=atol,T=duration,points=len(q),nfev=sol.nfev,
             bdg_growth=ev[k].real,bdg_frequency=ev[k].imag,dns_growth=growth,dns_frequency=frequency,
             relative_frequency_error=abs(frequency/ev[k].imag-1) if abs(ev[k].imag)>1e-8 else 0.,
             relative_growth_error=abs(growth/ev[k].real-1) if ev[k].real>1e-8 else 0.,
             norm_drift=max(abs(N/N[0]-1)),hamiltonian_drift=max(abs(H/H[0]-1)),seconds=time.perf_counter()-start,file=f'dns_{label}.npz')
    print('DNS',row,flush=True);return row

def prepare_collision(L=200):
    target=np.load(DATA/'representative_stable.npz');eps=float(target['epsilon']);Nt,Ht=invariants(target['q'],eps)
    def pulse(b):
        lat=Lattice(b,0,1,L=L);q,res=lat.solve(eps,q0=al(lat.n,b,0));return lat.n,q
    bs=root_scalar(lambda b:2*invariants(pulse(b)[1],eps)[0]-Nt,bracket=(.995,1.005),xtol=1e-13).root
    n,q=pulse(bs);ql=np.roll(q,-20);qr=np.roll(q,20)
    ql[-20:]=0;qr[:20]=0
    def initial(alpha,phase):return ql*np.exp(1j*alpha*(n+20))+qr*np.exp(1j*(phase-alpha*(n-20)))
    ae=root_scalar(lambda a:invariants(initial(a,np.pi),eps)[1]-Ht,bracket=(0,.02),xtol=1e-14).root
    Ni,Hi=invariants(initial(ae,np.pi),eps)
    Rstar=geometry(target['n'],target['q'])['Rmol']
    d=dict(n=n,ql=ql,qr=qr,epsilon=eps,beta_s=bs,alpha_E=ae,target_N=Nt,target_H=Ht,N_mismatch=Ni-Nt,H_mismatch=Hi-Ht,Rstar=Rstar)
    np.savez_compressed(DATA/f'collision_initial_L{L}.npz',**d)
    return d

def residence(t,R,Rstar,width):
    # Linear interpolation of each tube-entry/exit event on the stored grid.
    low=Rstar-width;high=Rstar+width;intervals=[];active=None
    for j in range(len(t)-1):
        a=t[j];b=t[j+1];ra=R[j];rb=R[j+1]
        cuts=[a,b]
        if rb!=ra:
            for bound in [low,high]:
                f=(bound-ra)/(rb-ra)
                if 0<f<1:cuts.append(a+f*(b-a))
        cuts.sort()
        for x,y in zip(cuts[:-1],cuts[1:]):
            rm=ra+((x+y)/2-a)/(b-a)*(rb-ra)
            inside=low<=rm<=high
            if inside and active is None:active=x
            if not inside and active is not None:intervals.append((active,x));active=None
    if active is not None:intervals.append((active,t[-1]))
    return max((b-a for a,b in intervals),default=0.)

def collision(job):
    phase,alpha,L,rtol,atol,T,tag=job[:7]
    output_dt=job[7] if len(job)>7 else 2.
    d=np.load(DATA/f'collision_initial_L{L}.npz');n=d['n'];eps=float(d['epsilon']);Rstar=float(d['Rstar'])
    z0=d['ql']*np.exp(1j*alpha*(n+20))+d['qr']*np.exp(1j*(phase-alpha*(n-20)))
    ts=np.arange(0,T+output_dt/2,output_dt)
    start=time.perf_counter()
    sol=solve_ivp(rhs,(0,T),z0,args=(1.,eps),method='DOP853',rtol=rtol,atol=atol,t_eval=ts)
    if not sol.success:raise RuntimeError(sol.message)
    N,H=invariants_series(sol.y,eps);rho=np.log1p(abs(sol.y)**2);left=n<0;right=n>0
    nl=rho[left].sum(axis=0);nr=rho[right].sum(axis=0)
    XL=(n[left,None]*rho[left]).sum(axis=0)/nl;XR=(n[right,None]*rho[right]).sum(axis=0)/nr;R=XR-XL
    # Core phases are recorded only as a secondary diagnostic, at the largest-amplitude site in each half.
    leftids=np.where(left)[0];rightids=np.where(right)[0]
    jl=leftids[np.argmax(abs(sol.y[left]),axis=0)];jr=rightids[np.argmax(abs(sol.y[right]),axis=0)]
    phasecore=np.angle(sol.y[jr,np.arange(len(ts))]*np.conj(sol.y[jl,np.arange(len(ts))]))
    filename=f'collision_{tag}.npz'
    np.savez_compressed(DATA/filename,t=ts,Rmol=R,XL=XL,XR=XR,N=N,H=H,deltaN=nr-nl,phasecore=phasecore,phase=phase,alpha=alpha,
                        epsilon=eps,Rstar=Rstar,n=n,snapshot_t=ts[::250],snapshots=sol.y[:,::250])
    row=dict(tag=tag,phase=phase,alpha=alpha,L=L,rtol=rtol,atol=atol,T=T,output_dt=output_dt,Rmin=min(R),Rfinal=R[-1],
             Tres1=residence(ts,R,Rstar,1),Tres2=residence(ts,R,Rstar,2),Tres3=residence(ts,R,Rstar,3),
             inside_final_1=abs(R[-1]-Rstar)<=1,inside_final_2=abs(R[-1]-Rstar)<=2,inside_final_3=abs(R[-1]-Rstar)<=3,
             norm_drift=max(abs(N/N[0]-1)),hamiltonian_drift=max(abs(H/H[0]-1)),nfev=sol.nfev,seconds=time.perf_counter()-start,file=filename)
    print('COLLISION',tag,'Rmin',row['Rmin'],'Tres2',row['Tres2'],'Rfinal',row['Rfinal'],'sec',row['seconds'],flush=True)
    return row

def main(mode):
    if mode=='dns':
        rows=[mode_dns(k) for k in ['stable0','stable1','stable2','real','HH']]
        rows += [mode_dns('stable0',rtol=2e-12,atol=2e-14),mode_dns('real',rtol=2e-12,atol=2e-14),
                 mode_dns('HH',rtol=2e-12,atol=2e-14),mode_dns('stable0',amplitude=5e-7)]
        dump('dns_benchmarks.csv',rows)
    elif mode=='pilot':
        d=prepare_collision(200);jsdump('collision_matching.json',{k:v for k,v in d.items() if k not in ['n','ql','qr']})
        rows=[]
        for phase,a,tag in [(np.pi,d['alpha_E'],'matched'),(np.pi,.0062,'residence'),(np.pi,.01,'fast')]:
            rows.append(collision((phase,a,200,1e-9,1e-11,5000,tag)))
        dump('collision_representatives.csv',rows)
    elif mode=='scan':
        prepare_collision(200)
        jobs=[]
        for ip,phase in enumerate(np.linspace(0,np.pi,9)):
            for ia,alpha in enumerate(np.arange(.0045,.0100001,.0005)):
                jobs.append((phase,alpha,200,1e-9,1e-11,5000,f'grid_p{ip}_a{ia}'))
        with ProcessPoolExecutor(max_workers=4) as pool:
            rows=list(pool.map(collision,jobs))
        dump('collision_scan.csv',rows)
    elif mode=='refine':
        prepare_collision(200);prepare_collision(400)
        jobs=[(np.pi,a,200,1e-9,1e-11,5000,f'refine_{j}') for j,a in enumerate(np.linspace(.0059,.0065,13))]
        with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(collision,jobs))
        dump('collision_refinement.csv',rows)
        best=max(rows,key=lambda r:r['Tres2']);match=json.loads((DATA/'collision_matching.json').read_text())['alpha_E']
        jobs=[(np.pi,best['alpha'],200,2e-11,2e-13,10000,'long_tight'),
              (np.pi,best['alpha'],400,2e-11,2e-13,5000,'large_domain'),
              (np.pi,match,400,2e-11,2e-13,5000,'matched_large')]
        with ProcessPoolExecutor(max_workers=3) as pool:checks=list(pool.map(collision,jobs))
        dump('collision_sensitivity.csv',checks)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['dns','pilot','scan','refine']);main(parser.parse_args().mode)
