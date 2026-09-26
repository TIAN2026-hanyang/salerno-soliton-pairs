"""Produce LaTeX tables from CSV files without hand-transcribed numbers."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
BASE=Path(__file__).resolve().parents[1];D=BASE/'data';O=BASE/'source'/'generated';O.mkdir(exist_ok=True)
def texnum(x,d=4):
    if x==0:return '0'
    return f'{x:.{d}g}'
def sci(x,d=3):
    if x==0:return '$0$'
    a,e=f'{x:.{d}e}'.split('e');return '$'+a+r'\times10^{'+str(int(e))+'}$'
def table(name,caption,headers,rows,alt,fmt=None,long=False):
    fmt=fmt or 'r'*len(headers);env='longtable' if long else 'tabular'
    s='\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\n'
    if long:
        s+='\\begin{longtable}{'+fmt+'}\n\\caption{'+caption+'}\\label{tab:'+name+'}\\\\\n\\toprule\n'
        s+=' & '.join(headers)+r' \\'+'\n\\midrule\\endfirsthead\n\\multicolumn{'+str(len(headers))+'}{c}{Table \\thetable{} continued}\\\\\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\\endhead\n\\bottomrule\\endfoot\n'
    else:
        s+='\\begin{table}[!htbp]\\centering\n\\caption{'+caption+'}\\label{tab:'+name+'}\n\\begin{tabular}{'+fmt+'}\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'
    s+='\n'.join(' & '.join(str(x) for x in row)+r' \\' for row in rows)+'\n'
    if long:s+='\\end{longtable}\n'
    else:s+='\\bottomrule\\end{tabular}\n\\par\\smallskip\\noindent\\textit{Alt text:} '+alt+'\n\\end{table}\n'
    if long:s+='\\noindent\\textit{Alt text:} '+alt+'\n'
    s+='\\par\\endgroup\n';(O/(name+'.tex')).write_text(s)

f=pd.read_csv(D/'folds_80.csv');th=pd.read_csv(D/'fredholm_prefactors.csv')
table('prefactors',r'Single-pulse predictions, obtained without fold fitting.',
      [r'$\beta$',r'$\sigma$',r'$x_*$',r'$C_\sigma$',r'$K(\beta)$'],
      [[f'{r.beta:g}',f'{r.sigma:+.0f}',f'{r.xstar:.9f}',f'{r.C:.9g}',f'{r.K:.9g}'] for r in th.itertuples()],
      'Ten phase and width combinations give distinct fold shifts and positive prefactors.')
rows=[]
for beta in sorted(f.beta.unique()):
    for m in sorted(f[f.beta==beta].m.unique()):
        a=f[(f.beta==beta)&(f.m==m)&(f.sigma==1)].iloc[0];b=f[(f.beta==beta)&(f.m==m)&(f.sigma==-1)].iloc[0]
        rows.append([f'{beta:g}',m,sci(a.epsilon,7),f'{a.rho:.8f}',sci(b.epsilon,7),f'{b.rho:.8f}'])
table('folds',r'All 80 numerical folds. Each row contains both phases; $\rho=\varepsilon_{\rm fold}/(C_\sigma e^{-\beta m})$.',
      [r'$\beta$','$m$',r'$\varepsilon_0$',r'$\rho_0$',r'$\varepsilon_\pi$',r'$\rho_\pi$'],rows,
      'Forty rows list equal-phase and opposite-phase thresholds, totaling eighty folds. Prediction ratios approach one.',long=True)
c=pd.read_csv(D/'fold_convergence.csv');rows=[]
for r in c.itertuples():
    baseline=c[(c.beta==r.beta)&(c.m==r.m)&(c.sigma==r.sigma)&(c.padding==32)].iloc[0].epsilon
    rows.append([f'{r.beta:g}',f'{r.sigma:+.0f}',r.m,r.padding,sci(r.solver_xtol,0),sci(r.epsilon,8),sci(abs(r.epsilon/baseline-1),2)])
table('foldconv',r'Fold sensitivity to tail padding $P$ and solver termination tolerance. The last column compares with $P=32$.',
      [r'$\beta$',r'$\sigma$','$m$','$P$',r'\texttt{xtol}',r'$\varepsilon_{\rm fold}$','Rel. change'],rows,
      'Three representative folds are recomputed with three domains and two solver tolerances.')
rows=[]
for (b,s),r in f.groupby(['beta','sigma']):rows.append([f'{b:g}',f'{s:+d}',sci(r.residual.max()),sci(r.null_residual.max()),sci(abs(r.transversality).min()),sci(abs(r.quadratic).min())])
table('nondeg',r'Residual and nondegeneracy checks over each eight-fold family.',
      [r'$\beta$',r'$\sigma$',r'$\max\|F\|_\infty$',r'$\max\|Jv\|_\infty$',r'$\min|w^TF_\varepsilon|$',r'$\min|w^TF_{\phi\phi}[v,v]|$'],rows,
      'Each family has small equation residuals and nonzero transversality and quadratic fold coefficients.')
s=pd.read_csv(D/'spectral_grid.csv');rows=[]
for r in s.itertuples():rows.append([f'{r.beta:g}',f'{r.eta:g}',r.m,sci(r.exchange,6),sci(r.pinning,6),sci(r.stretch,6),sci(max(r.exchange_error,r.pinning_error,r.stretch_error),2)])
table('spectralall',r'All 75 fixed-$\eta$ states. The last column is the largest relative error of the three leading frequency predictions.',
      [r'$\beta$',r'$\eta$','$m$',r'$\Omega_{\rm ex}$',r'$\Omega_{\rm PN}$',r'$\Omega_{\rm str}$',r'$E_{\max}$'],rows,
      'Seventy-five rows give three frequencies per state and the error of the explicit leading coefficients.',long=True)
k=pd.read_csv(D/'spectral_exponents.csv');rows=[]
for (_,eta),rr in k.groupby(['beta','eta']):
    r=rr.loc[rr.m.idxmax()];ss=s[(s.beta==r.beta)&(s.eta==eta)&(s.m==r.m_next)].iloc[0]
    rows.append([f'{r.beta:g}',f'{eta:g}',f'{r.m:.0f},{r.m_next:.0f}',f'{r.exchange:.7f}',f'{r.pinning:.7f}',f'{r.stretch:.7f}',sci(max(ss.exchange_error,ss.pinning_error,ss.stretch_error),2)])
table('exponents',r'Largest adjacent-separation pair for every width and leading-fold normalization. The final column is the maximum leading-frequency error at the larger separation.',
      [r'$\beta$',r'$\eta$','Pair',r'$\kappa_{\rm ex}$',r'$\kappa_{\rm PN}$',r'$\kappa_{\rm str}$',r'$E_{\max}$'],rows,
      'All fifteen width-normalization combinations approach the predicted exponent beta.')
c=pd.read_csv(D/'spectral_convergence.csv')
table('specconv',r'Domain convergence for the representative stable state.',
      ['$P$','Sites',r'$\Omega_{\rm ex}$',r'$\Omega_{\rm PN}$',r'$\Omega_{\rm str}$','Eigen-residual'],
      [[r.padding,r.points,f'{r.exchange:.12f}',f'{r.pinning:.12f}',f'{r.stretch:.12f}',sci(r.eig_residual,2)] for r in c.itertuples()],
      'Three domains reproduce the three frequencies to the displayed precision apart from final digits.')
d=pd.read_csv(D/'daughter_branches.csv')
table('daughters',r'One asymmetric daughter for $\beta=1,m=10$; reflection gives the daughter at $-A$ with the same spectrum.',
      ['$A$',r'$\varepsilon$',r'$R_{\rm mol}$',r'$\Delta N/N$',r'$\max\operatorname{Re}\lambda$'],
      [[f'{r.A:.3f}',f'{r.epsilon:.10f}',f'{r.Rmol:.8f}',sci(r.norm_asymmetry,3),f'{r.max_real:.8f}'] for r in d.itertuples()],
      'Sixteen asymmetric profiles have a nonzero center shift and a real unstable eigenvalue.')
h=pd.read_csv(D/'HH_convergence.csv')
table('hhconv',r'Hamiltonian--Hopf refinement using the discriminant of the two relevant squared eigenvalues.',
      ['$P$','Sites',r'$\varepsilon_{\rm HH}$',r'$\Omega_{\rm HH}$',r'$D(\varepsilon_{\rm HH}-10^{-7})$',r'$D(\varepsilon_{\rm HH}+10^{-7})$'],
      [[r.padding,r.points,f'{r.epsilon_HH:.12f}',f'{r.frequency:.12f}',sci(r.discriminant_below,2),sci(r.discriminant_above,2)] for r in h.itertuples()],
      'Three domains yield the same collision point and a positive-to-negative discriminant crossing.')
dns=pd.read_csv(D/'dns_benchmarks.csv')
table('dns',r'Time-domain benchmarks and sensitivity checks. Errors compare fitted frequencies and positive growth rates with BdG values; a dash denotes a quantity not fitted as a nonzero benchmark.',
      ['Mode','Amplitude','rtol',r'$E_\Omega$',r'$E_\gamma$',r'$\delta N$',r'$\delta H$'],
      [[r.kind,sci(r.amplitude,0),sci(r.rtol,0),sci(r.relative_frequency_error,2) if r.bdg_frequency>0 else '--',sci(r.relative_growth_error,2) if r.bdg_growth>1e-8 else '--',sci(r.norm_drift,2),sci(r.hamiltonian_drift,2)] for r in dns.itertuples()],
      'Nine integrations test three stable modes, two unstable modes, tighter tolerances, and smaller amplitude.',fmt='lrrrrrr')
cr=pd.read_csv(D/'collision_representatives.csv');cs=pd.read_csv(D/'collision_sensitivity.csv');co=pd.read_csv(D/'collision_output_check.csv')
rows=[]
for r in pd.concat([cr,cs,co[co.tag=='dense_output']]).itertuples():
    label={'matched':'Matched','residence':r'$\alpha=0.0062$','fast':r'$\alpha=0.0100$','long_tight':'Long/tight','large_domain':'Large/tight','matched_large':'Matched/large','dense_output':'Dense output'}[r.tag]
    rows.append([label,r.L,int(r.T),f'{r.output_dt:g}',sci(r.rtol,0),f'{r.Tres1:.3f}',f'{r.Tres2:.3f}',f'{r.Tres3:.3f}',f'{r.Rfinal:.3f}'])
table('collisionpi',r'Opposite-phase collision checks. Except for the matched and fast cases, $\alpha=0.0062$.',
      ['Case','$L$','$T$',r'$\Delta t$','rtol',r'$T_1$',r'$T_2$',r'$T_3$',r'$R(T)$'],rows,
      'The matched case and selected rebound case are consistent under domain, tolerance, and output-spacing changes.',fmt='lrrrrrrrr')
scan=pd.read_csv(D/'collision_scan.csv');ci=pd.read_csv(D/'collision_inphase_sensitivity.csv')
rr=pd.concat([scan[scan.tag=='grid_p0_a6'],co[co.tag=='coarse_max_check'],ci]).sort_values(['L','rtol'],ascending=[True,False]);rows=[]
for r in rr.itertuples():rows.append([r.L,sci(r.rtol,0),f'{r.output_dt:g}',f'{r.Tres2:.3f}',f'{r.Rfinal:.3f}',sci(r.norm_drift,2),sci(r.hamiltonian_drift,2)])
table('collisionzero',r'Sensitivity of the in-phase case $\alpha=0.0075$, all with $T=5000$. This residence peak is not converged.',
      ['$L$','rtol',r'$\Delta t$',r'$T_{\rm res}^{(2)}$',r'$R(T)$',r'$\delta N$',r'$\delta H$'],rows,
      'Residence changes substantially with domain and tolerance even when conserved quantities have small drift.')
print('TABLES COMPLETE')

cs=pd.read_csv(D/'softening_thresholds.csv');rows=[]
for beta,r in cs.groupby('beta'):
    a=r.iloc[0]
    rows.append([f'{beta:g}',f'{a.x_CS_lead:.9f}',f'{a.C_CS:.9g}',f'{a.eta_CS_lead:.9f}'])
table('csprefactors',r'Center-softening predictions evaluated from the single-pulse force.',
      [r'$\beta$',r'$x_{\rm cs}$',r'$C_{\rm cs}$',r'$\eta_{\rm cs}^{\rm lead}$'],rows,
      'All five predicted center-softening thresholds lie above their leading folds; the normalized gap increases over the sampled widths.')
rows=[]
for r in cs.itertuples():
    rows.append([f'{r.beta:g}',r.m,sci(r.epsilon_CS,7),sci(r.prediction,7),sci(r.relative_error,2),f'{r.eta_CS_numerical:.8f}'])
table('softening',r'All 40 center-softening zeros. The error is $|\varepsilon_{\rm cs}^{\rm num}/\varepsilon_{\rm cs}^{\rm lead}-1|$ and the final column uses the numerical fold.',
      [r'$\beta$','$m$',r'$\varepsilon_{\rm cs}^{\rm num}$',r'$\varepsilon_{\rm cs}^{\rm lead}$','Rel. error',r'$\widehat\eta_{\rm cs}$'],rows,
      'Forty zeros of the opposite-parity stationary operator approach the force-based softening prediction as separation increases.',long=True)
c=pd.read_csv(D/'softening_convergence.csv');rows=[]
for r in c.itertuples():
    ref=c[(c.beta==r.beta)&(c.m==r.m)&(c.padding==32)].iloc[0].epsilon_CS
    rows.append([f'{r.beta:g}',r.m,r.padding,sci(r.epsilon_CS,8),sci(abs(r.epsilon_CS/ref-1),2),sci(r.null_residual,2)])
table('csconv',r'Center-softening domain checks with solver termination tolerance $10^{-11}$. Changes are relative to padding 32.',
      [r'$\beta$','$m$','$P$',r'$\varepsilon_{\rm cs}^{\rm num}$','Rel. change',r'$\|Jv\|_\infty$'],rows,
      'Three representative center zeros agree across three padding values; changes are much smaller than the asymptotic errors.')
sr=pd.read_csv(D/'spectral_sum_rule_summary.csv');rows=[]
for r in sr.itertuples():
    rows.append([f'{r.beta:g}',f'{r.eta:g}',f'{r.m_first},{r.m_last}',f'{r.eta_num_first:.6f}',f'{r.eta_num_last:.6f}',sci(r.residual_first,3),sci(r.residual_last,3)])
table('sumrule',r'Leading and numerical fold normalizations and signed sum-rule residuals at the first and last separations. The full 75-state table is supplied as CSV.',
      [r'$\beta$',r'$\eta$','First,last',r'$\widehat\eta_{\rm first}$',r'$\widehat\eta_{\rm last}$',r'$\mathcal S_{\rm first}$',r'$\mathcal S_{\rm last}$'],rows,
      'Fifteen parameter families distinguish the exact numerical fold offset from the leading normalization and show smaller sum-rule residuals at their largest separations.')
print('V8 ADDITIONAL TABLES COMPLETE')
