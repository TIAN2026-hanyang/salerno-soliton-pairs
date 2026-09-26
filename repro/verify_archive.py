"""Check archived values against manuscript claims, without a parameter scan."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
import scipy
from model import stationary_F, stationary_J, fold_theory, frequency_theory, Lattice, identify_modes
from run_reduced_checks import secondary_zero

BASE = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=BASE/'verification_output')
    p.add_argument('--smoke', action='store_true', help='Also solve one fold, one spectrum and one secondary zero.')
    a = p.parse_args()
    out = a.output_dir.resolve()
    if any(out == x or out.is_relative_to(x) for x in [BASE/'data', BASE/'repro', BASE/'figures', BASE/'provenance']):
        p.error('Output must not overwrite archived inputs.')
    out.mkdir(parents=True, exist_ok=True)
    D = BASE/'data'
    checks = []
    def check(name, passed, observed, evidence):
        checks.append(dict(check=name, passed=bool(passed), observed=observed, evidence=evidence))
    original = json.loads((BASE/'provenance/ORIGINAL_DATA_HASHES.json').read_text())
    bad = [r['file'] for r in original if not (BASE/r['file']).exists() or
           hashlib.sha256((BASE/r['file']).read_bytes()).hexdigest() != r['sha256']]
    check('Original data bytes preserved', not bad, {'count':len(original), 'mismatches':bad}, 'provenance/ORIGINAL_DATA_HASHES.json')
    if (BASE/'MANIFEST_SHA256.txt').exists():
        bad = []
        for line in (BASE/'MANIFEST_SHA256.txt').read_text().splitlines():
            sha, name = line.split('  ', 1)
            if not (BASE/name).exists() or hashlib.sha256((BASE/name).read_bytes()).hexdigest() != sha:
                bad.append(name)
        check('Complete manifest', not bad, bad, 'MANIFEST_SHA256.txt')
    f = pd.read_csv(D/'folds_80.csv'); s = pd.read_csv(D/'spectral_grid.csv')
    z = pd.read_csv(D/'softening_thresholds.csv'); u = pd.read_csv(D/'spectral_sum_rule.csv')
    ranges = {0.6:(24,31),0.75:(17,24),1.0:(10,17),1.25:(8,15),1.5:(6,13)}
    grid_ok = len(f)==80 and len(s)==75 and len(z)==40
    for beta,(lo,hi) in ranges.items():
        for sig in [1,-1]:
            grid_ok &= sorted(f[(f.beta==beta)&(f.sigma==sig)].m.tolist()) == list(range(lo,hi+1))
        grid_ok &= sorted(z[z.beta==beta].m.tolist()) == list(range(lo,hi+1))
        for eta in [1.25,2,4]:
            grid_ok &= sorted(s[(s.beta==beta)&(s.eta==eta)].m.tolist()) == list(range(hi-4,hi+1))
    check('Table C.1 parameter grids',grid_ok,[len(f),len(s),len(z)],'folds_80.csv; spectral_grid.csv; softening_thresholds.csv')
    check('Fold residual bounds',f.residual.max()<9e-13 and f.null_residual.max()<5e-13,
          [f.residual.max(),f.null_residual.max()],'folds_80.csv')
    check('Fold nondegeneracy',(f.transversality!=0).all() and (f.quadratic!=0).all(),
          [f.transversality.abs().min(),f.quadratic.abs().min()],'folds_80.csv')
    for filename,col,keys,bound in [('fold_convergence.csv','epsilon',['beta','sigma','m'],4e-9),
                                  ('softening_convergence.csv','epsilon_CS',['beta','m'],3.5e-9)]:
        c = pd.read_csv(D/filename); delta=[]
        for _,g in c.groupby(keys):
            ref=g[g.padding==32][col].iloc[0]; delta.extend(abs(g[col]/ref-1).tolist())
        check(filename,len(c)==9 and max(delta)<bound,{'rows':len(c),'max_relative_change':max(delta)},filename)
    c = pd.read_csv(D/'spectral_convergence.csv')
    rounded={k:[f'{x:.12f}' for x in c[k]] for k in ['exchange','pinning','stretch']}
    check('Spectral padding: 12 decimal places',len(c)==3 and all(len(set(x))==1 for x in rounded.values()),rounded,'spectral_convergence.csv')
    check('Secondary-zero diagnostics',z.residual.max()<4e-15 and z.null_residual.max()<5e-15 and
          z.center_tangent_overlap.min()>=0.999973 and (z.curvature_below<0).all() and (z.curvature_above>0).all(),
          [z.residual.max(),z.null_residual.max(),z.center_tangent_overlap.min()],'softening_thresholds.csv')
    profile_rows=[]
    for fn in sorted(list(D.glob('fold_b*.npz'))+list(D.glob('softening_b*.npz'))):
        d=np.load(fn,allow_pickle=False);eps=float(d['epsilon'] if 'epsilon' in d.files else d['epsilon_CS']);beta=float(d['beta'])
        r=float(np.max(abs(stationary_F(d['q'],beta,eps))));n=float(np.max(abs(stationary_J(d['q'],beta,eps)@d['v'])))
        profile_rows.append(dict(file=fn.name,stationary_residual=r,null_residual=n))
    pr=pd.DataFrame(profile_rows);pr.to_csv(out/'profile_residuals.csv',index=False)
    check('120 stored profiles satisfy F=0 and Jv=0',len(pr)==120 and pr.stationary_residual.max()<9e-13 and pr.null_residual.max()<5e-13,
          [len(pr),pr.stationary_residual.max(),pr.null_residual.max()],'fold_b*.npz; softening_b*.npz')
    check('225 Krein signs',(s.exchange_krein<0).all() and (s.pinning_krein>0).all() and (s.stretch_krein>0).all(),225,'spectral_grid.csv')
    r=(s.stretch**2-s.pinning**2-s.exchange**2)/s.stretch**2
    last=u.loc[u.groupby(['beta','eta']).m.idxmax()].sum_residual_absolute.max()
    check('Sum rule: stored values and printed bounds',np.max(abs(r-u.sum_residual_signed))<1e-12 and abs(r).max()<1.03e-3 and last<3.76e-5,
          [abs(r).max(),last],'spectral_grid.csv; spectral_sum_rule.csv')
    coeff=pd.read_csv(D/'fredholm_prefactors.csv');cerr=0.
    for row in coeff.itertuples():
        x,C,K=fold_theory(row.beta,row.sigma)
        cerr=max(cerr,abs(x-row.xstar),abs(C/row.C-1),abs(K/row.K-1))
    check('Table A.1 prefactors independently evaluated',len(coeff)==10 and cerr<1e-9,cerr,'fredholm_prefactors.csv; model.fold_theory')
    dns=pd.read_csv(D/'dns_benchmarks.csv');dns=dns[dns.kind!='HH'];rows=[];agreement=True
    for row in dns.itertuples():
        d=np.load(D/row.file,allow_pickle=False);t=d['t'];v=d['c'];ev=d['lambda_bdg'];sl=slice(30,-30)
        growth=np.polyfit(t[sl],np.log(abs(v))[sl],1)[0];freq=np.polyfit(t[sl],np.unwrap(np.angle(v))[sl],1)[0]
        err=abs(growth/ev.real-1) if row.kind=='real' else abs(freq/ev.imag-1)
        nd=max(abs(d['N']/d['N'][0]-1));hd=max(abs(d['H']/d['H'][0]-1))
        agreement &= len(t)==2401 and abs(growth-row.dns_growth)<1e-13 and abs(freq-row.dns_frequency)<1e-13
        agreement &= abs(nd-row.norm_drift)<1e-18 and abs(hd-row.hamiltonian_drift)<1e-18
        rows.append(dict(file=row.file,relative_fit_error=err,norm_drift=nd,hamiltonian_drift=hd))
    q=pd.DataFrame(rows);q.to_csv(out/'seven_dns_refits.csv',index=False)
    check('Seven non-HH time-domain checks',len(q)==7 and agreement and q.relative_fit_error.max()<8e-8 and
          q[['norm_drift','hamiltonian_drift']].max().max()<3e-15,
          [len(q),q.relative_fit_error.max(),q[['norm_drift','hamiltonian_drift']].max().max()],'dns_benchmarks.csv and seven NPZ trajectories')
    values=json.loads((D/'representative_values.json').read_text());daughters=pd.read_csv(D/'daughter_branches.csv')
    coef=(daughters.epsilon.iloc[0]-values['epsilon_SB'])/daughters.A.iloc[0]**2
    check('Daughter branch data',len(daughters)==16 and (daughters.max_real>0).all() and abs(coef-0.1842)<5e-5,
          {'count':len(daughters),'A_min':daughters.A.min(),'A_max':daughters.A.max(),'quadratic_estimate':coef},'daughter_branches.csv; representative_values.json')
    if a.smoke:
        lat=Lattice(1,10,-1);qf,vf,rf=lat.fold();_,_,_,rz=secondary_zero(1,10)
        lat=Lattice(1,12,-1);q0,res=lat.solve(values['epsilon_stable']);m=identify_modes(lat,q0,values['epsilon_stable'])
        errs=[abs(rf['epsilon']/values['epsilon_fold']-1),abs(rz['epsilon_CS']/values['epsilon_SB']-1),float(max(abs(m['frequencies']/np.array(values['frequencies'])-1)))]
        check('Representative fresh fold / zero / spectrum solves',max(errs)<1e-7,errs,'model; run_reduced_checks.secondary_zero')
    result=dict(all_passed=all(x['passed'] for x in checks),checks=checks,
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                scope='Stored-data verification; optional representative solves. Full parameter scans not rerun.')
    (out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['# Numerical archive verification','','| Check | Result | Observed |','|---|---|---|']
    lines += [f"| {r['check']} | {'PASS' if r['passed'] else 'FAIL'} | {r['observed']} |" for r in checks]
    (out/'verification.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['all_passed']:sys.exit(1)

if __name__=='__main__':
    main()
