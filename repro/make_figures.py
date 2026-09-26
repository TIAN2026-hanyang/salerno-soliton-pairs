"""Generate every vector figure directly from the supplied numerical outputs."""
from pathlib import Path
import json
import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from model import *

BASE=Path(__file__).resolve().parents[1];D=BASE/'data';OUT=BASE/'source'/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':9,'axes.titlesize':9,
                     'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,'lines.linewidth':1.3,
                     'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.pad_inches':.03})
colors=['#286CA6','#D16A28','#259273','#8B5BA5','#A24459'];names=['exchange','pinning','stretch'];lab=['Exchange','PN pinning','Stretch']
folds=pd.read_csv(D/'folds_80.csv');spec=pd.read_csv(D/'spectral_grid.csv');exps=pd.read_csv(D/'spectral_exponents.csv')
rep=np.load(D/'representative_stable.npz');vals=json.loads((D/'representative_values.json').read_text())
def panels(axes):
    for letter,ax in zip('abcdefghijkl',np.asarray(axes).ravel()):
        ax.text(0,1.04,'('+letter+')',transform=ax.transAxes,weight='bold',va='bottom')
        ax.grid(alpha=.13,lw=.5)
def save(fig,name):
    for fmt in ['pdf','png']:
        buffer=io.BytesIO();fig.savefig(buffer,format=fmt,dpi=160,bbox_inches='tight')
        target=OUT/(name+'.'+fmt);temporary=OUT/(name+'.'+fmt+'.tmp')
        with temporary.open('wb') as f:
            f.write(buffer.getvalue());f.flush();os.fsync(f.fileno())
        temporary.replace(target)
    plt.close(fig)

fig,ax=plt.subplots(2,2,figsize=(7,4.1),layout='constrained');panels(ax)
n=np.arange(-6,7)
for X,c in zip([0,.25,.5],colors):ax[0,0].plot(n,al(n,1,X),'o-',ms=2.5,color=c,label=f'$X={X:g}$')
ax[0,0].set(xlabel='Lattice site $n$',ylabel='$Q_n(X)$');ax[0,0].legend(frameon=False)
x=np.linspace(-.5,.5,501);M=np.array([melnikov(y,1) for y in x])
ax[0,1].plot(x,M,color=colors[0]);ax[0,1].axhline(0,c='gray',lw=.6);ax[0,1].set(xlabel='Single-pulse center $X$',ylabel='$M(X;1)$')
xx=np.arange(-9,10);ql=al(xx,1,-6);qr=al(xx,1,6)
ax[1,0].semilogy(xx,ql,'o-',ms=2,color=colors[0],label='Left pulse');ax[1,0].semilogy(xx,qr,'s-',ms=2,color=colors[1],label='Right pulse')
ax[1,0].axvline(0,color='gray',ls=':',lw=.8);ax[1,0].set(xlabel='$n$',ylabel='Tail amplitude',ylim=(1e-7,2));ax[1,0].legend(frameon=False)
xx=np.linspace(.001,.499,499);C=fold_theory(1,-1)[1];K=fold_theory(1,-1)[2]
ax[1,1].plot(xx,[2*C*melnikov(-v,1) for v in xx],label=r'$\eta C_\pi M(-x)$',color=colors[0])
ax[1,1].plot(xx,K*np.exp(-2*xx),label=r'$K\exp(-2x)$',color=colors[1]);ax[1,1].set(xlabel='Half-shift $x$',ylabel='Scaled force',title=r'Out-of-phase balance, $\eta=2$');ax[1,1].legend(frameon=False)
save(fig,'fig1')

fig,ax=plt.subplots(1,2,figsize=(7,2.55),layout='constrained');panels(ax)
for j,sigma in enumerate([1,-1]):
    lat=Lattice(1,12,sigma);eps=2*fold_theory(1,sigma)[1]*np.exp(-12);q,res=lat.solve(eps);x=branch_x(1,sigma,2)
    ax[j].plot(lat.n,q,'o',ms=3,color=colors[j],label='Stationary lattice solution')
    ax[j].plot(lat.n,lat.guess(x),'--',color='black',lw=1,label='Leading AL pair')
    ax[j].set(xlim=(-12,12),xlabel='$n$',ylabel=r'$\phi_n$',title=rf'$\sigma={sigma:+d}$, $\varepsilon={eps:.4g}$');ax[j].legend(frameon=False,loc='lower left')
save(fig,'fig2')

fig,ax=plt.subplots(1,2,figsize=(7,2.8),layout='constrained');panels(ax)
for ib,beta in enumerate(sorted(folds.beta.unique())):
    for sigma,marker,style in [(1,'o','-'),(-1,'s','--')]:
        r=folds[(folds.beta==beta)&(folds.sigma==sigma)].sort_values('m')
        ax[0].semilogy(r.m,r.epsilon,marker=marker,ms=3,color=colors[ib],ls=style,label=rf'$\beta={beta:g}$' if sigma==1 else None)
        kk=-np.diff(np.log(r.epsilon))/beta
        ax[1].plot(r.m.values[:-1],kk,marker=marker,ms=3,color=colors[ib],ls=style)
ax[0].set(xlabel='Integer separation $m$',ylabel=r'$\varepsilon_{\rm fold}$');ax[0].legend(frameon=False,ncol=2)
ax[1].axhline(1,c='black',lw=.8);ax[1].set(xlabel='$m$ (pair $m,m+1$)',ylabel=r'$\kappa_{\rm fold}/\beta$');ax[1].text(.03,.04,'Circles: in phase\nSquares: out of phase',transform=ax[1].transAxes,fontsize=7)
save(fig,'fig3')

fig,ax=plt.subplots(2,2,figsize=(7,4.2),layout='constrained');panels(ax)
for sigma,col in [(1,colors[0]),(-1,colors[1])]:
    xf,C,K=fold_theory(1,sigma);xx=np.linspace(-.49,-.01,400) if sigma==1 else np.linspace(.01,.49,400)
    cc=np.array([-sigma*K*np.exp(-2*z)/melnikov(-z,1) for z in xx]);ax[0,0].plot(xx,cc/C,color=col,label=r'$\sigma=+1$' if sigma==1 else r'$\sigma=-1$');ax[0,0].plot(xf,1,'o',color=col,ms=4)
ax[0,0].set(xlabel='$x$',ylabel=r'$C_\sigma(x)/C_\sigma(x_*)$',ylim=(.9,3.5));ax[0,0].legend(frameon=False)
th=pd.read_csv(D/'fredholm_prefactors.csv')
for sig,marker,label in [(1,'o',r'$C_0$'),(-1,'s',r'$C_\pi$')]:
    r=th[th.sigma==sig];ax[0,1].semilogy(r.beta,r.C,marker+'-',ms=4,label=label)
ax[0,1].set(xlabel=r'$\beta$',ylabel='Fold prefactor');ax[0,1].legend(frameon=False)
for ib,beta in enumerate(sorted(folds.beta.unique())):
    for sigma,marker,style in [(1,'o','-'),(-1,'s','--')]:
        r=folds[(folds.beta==beta)&(folds.sigma==sigma)]
        ax[1,0].plot(beta*r.m,r.rho,marker=marker,ms=3,ls=style,color=colors[ib])
        ax[1,1].semilogy(beta*r.m,r.relative_error,marker=marker,ms=3,ls=style,color=colors[ib],label=rf'$\beta={beta:g}$' if sigma==1 else None)
ax[1,0].axhline(1,c='black',lw=.7);ax[1,0].set(xlabel=r'$\beta m$',ylabel=r'$\rho=\varepsilon_{\rm fold}/(C_\sigma e^{-\beta m})$')
ax[1,1].set(xlabel=r'$\beta m$',ylabel=r'$|\rho-1|$');ax[1,1].legend(frameon=False,ncol=2,loc='upper right')
save(fig,'fig4')

fig,ax=plt.subplots(2,2,figsize=(7,4.15),layout='constrained');panels(ax)
cs=pd.read_csv(D/'softening_thresholds.csv')
for ib,beta in enumerate(sorted(cs.beta.unique())):
    r=cs[cs.beta==beta].sort_values('m');col=colors[ib]
    ax[0,0].plot(beta*r.m,r.eta_CS_numerical,'o-',ms=3,color=col,label=rf'$\beta={beta:g}$')
    ax[0,0].plot(beta*r.m,np.full(len(r),r.eta_CS_lead.iloc[0]),'--',lw=.8,color=col)
    ax[0,1].semilogy(beta*r.m,r.relative_error,'o-',ms=3,color=col)
ax[0,0].set(xlabel=r'$\beta m$',ylabel=r'$\varepsilon_{\rm cs}^{\rm num}/\varepsilon_{\rm fold}^{\rm num}$')
ax[0,0].legend(frameon=False,ncol=2,fontsize=6.5)
ax[0,1].set(xlabel=r'$\beta m$',ylabel=r'$|\varepsilon_{\rm cs}^{\rm num}/\varepsilon_{\rm cs}^{\rm lead}-1|$')
b=pd.read_csv(D/'branches_m10.csv')
for name,style in [('main','-'),('companion','--')]:
    r=b[b.branch==name];ax[1,0].plot(r.epsilon,r.Rmol,color='0.75',ls=style,lw=1)
    for stable,col,label in [(True,colors[0],'Spectrally stable'),(False,colors[1],'Unstable')]:
        rr=r[(r.max_real<1e-6)==stable];ax[1,0].scatter(rr.epsilon,rr.Rmol,s=9,c=col,label=label if name=='main' else None)
ax[1,0].axvline(vals['epsilon_SB'],ls=':',c='black',lw=.8)
ax[1,0].set(xlim=(.0092,.02),xlabel=r'$\varepsilon$',ylabel=r'$R_{\rm mol}$',title=r'$\beta=1$, $m=10$')
ax[1,0].legend(frameon=False,loc='lower left',fontsize=6.5)
d=pd.read_csv(D/'daughter_branches.csv')
for sign in [-1,1]:ax[1,1].plot(d.epsilon,sign*d.A,'o-',ms=3,color=colors[1])
ax[1,1].plot([.0093,vals['epsilon_SB']],[0,0],c=colors[1]);ax[1,1].plot([vals['epsilon_SB'],.018],[0,0],c=colors[0])
ax[1,1].set(xlim=(.0093,.018),xlabel=r'$\varepsilon$',ylabel='Center shift $A$',title='Asymmetric daughters')
save(fig,'fig6')

fig,ax=plt.subplots(2,3,figsize=(7,4.3),layout='constrained');panels(ax)
for j in range(3):
    mode=rep['modes'][j]/np.linalg.norm(rep['modes'][j]);tan=rep['tangents'][j]/np.linalg.norm(rep['tangents'][j]);
    ax[0,j].plot(rep['n'],mode,'o',ms=2.7,color=colors[j],label='BdG component');ax[0,j].plot(rep['n'],tan,'-',lw=.8,c='black',label='AL tangent')
    ax[0,j].set(xlim=(-11,11),xlabel='$n$',ylabel='Unit component',title=lab[j]);ax[0,j].set_xticks([-10,0,10])
mat=rep['overlaps'];im=ax[1,0].imshow(mat,cmap='Blues',vmin=0,vmax=1)
for i in range(3):
    for j in range(3):
        t=f'{mat[i,j]:.7f}' if i==j else (f'{mat[i,j]:.5f}' if mat[i,j]>.0001 else '$<10^{-10}$');ax[1,0].text(j,i,t,ha='center',va='center',fontsize=6,color='white' if i==j else 'black')
ax[1,0].set_xticks([0,1,2],['Ex.','PN','Str.']);ax[1,0].set_yticks([0,1,2],['Ex.','PN','Str.']);ax[1,0].set(xlabel='AL tangent',ylabel='BdG component',title='Shape overlaps');ax[1,0].grid(False)
r=spec[(spec.beta==1)&(spec.eta==2)]
for j,name in enumerate(names):
    ax[1,1].plot(r.m,r[name]*np.exp(r.m/2),'o-',ms=3,color=colors[j],label=lab[j]);ax[1,1].plot(r.m,r[name+'_theory']*np.exp(r.m/2),'--',color=colors[j],lw=.8)
ax[1,1].set(xlabel='$m$',ylabel=r'$\Omega_j e^{m/2}$',title=r'$\beta=1$, $\eta=2$');ax[1,1].legend(frameon=False,fontsize=6)
for j,name in enumerate(names):
    rr=exps.loc[exps[exps.eta==2].groupby('beta').m.idxmax()];ax[1,2].plot(rr.beta,rr[name]/rr.beta,'o-',ms=3,color=colors[j],label=lab[j])
ax[1,2].axhline(1,c='black',lw=.6);ax[1,2].ticklabel_format(axis='y',style='plain',useOffset=False);ax[1,2].set(xlabel=r'$\beta$',ylabel=r'$\kappa_j/\beta$',title='Largest pair')
save(fig,'fig5')

fig,ax=plt.subplots(2,2,figsize=(7,4.2),layout='constrained');panels(ax)
for j in range(3):
    z=np.load(D/f'dns_stable{j}_a1e-06_tol1e-10.npz');t=z['t'];freq=z['lambda_bdg'].imag;c=z['c']/z['c'][0]
    select=t<400;ax[0,0].plot(t[select],c.real[select]+2.3*j,color=colors[j]);ax[0,0].plot(t[select][::20],np.cos(freq*t[select][::20])+2.3*j,'o',ms=2,color='black')
ax[0,0].set(xlabel='$t$',ylabel='Normalized modal signal + offset');ax[0,0].set_yticks([0,2.3,4.6],['Ex.','PN','Str.'])
for kind,col in [('real',colors[0])]:
    z=np.load(D/f'dns_{kind}_a1e-06_tol1e-10.npz');t=z['t'];y=np.log(abs(z['c']/z['c'][0]));ax[0,1].plot(t,y,color=col,label=kind.upper() if kind=='HH' else 'Real pair');ax[0,1].plot(t[::120],z['lambda_bdg'].real*t[::120],'o',ms=2,c='black')
ax[0,1].set(xlabel='$t$',ylabel=r'$\ln|c(t)/c(0)|$');ax[0,1].legend(frameon=False)
for tag,col,label in [('matched',colors[0],'Invariant matched'),('residence',colors[1],r'$\alpha=0.0062$'),('fast',colors[2],r'$\alpha=0.0100$')]:
    z=np.load(D/f'collision_{tag}.npz');ax[1,0].plot(z['t'],z['Rmol'],color=col,label=label)
ax[1,0].axhspan(vals['Rmol']-2,vals['Rmol']+2,color='gray',alpha=.2);ax[1,0].set(xlabel='$t$',ylabel=r'$R_{\rm mol}(t)$',ylim=(5,105));ax[1,0].legend(frameon=False,fontsize=6.5)
r=pd.read_csv(D/'collision_refinement.csv')
for j,w in enumerate([1,2,3]):ax[1,1].plot(r.alpha,r[f'Tres{w}'],'o-',ms=3,color=colors[j],label=rf'$d={w}$')
ax[1,1].set(xlabel=r'Incoming phase kick $\alpha$',ylabel=r'$T_{\rm res}^{(d)}$',title=r'$\Delta\Phi=\pi$');ax[1,1].legend(frameon=False)
save(fig,'fig7')

fig,ax=plt.subplots(1,3,figsize=(7,2.4),layout='constrained');panels(ax)
for j,name in enumerate(names):
    for ib,beta in enumerate(sorted(spec.beta.unique())):
        r=spec[(spec.beta==beta)&(spec.eta==2)];ax[j].semilogy(beta*r.m,r[name+'_error'],'o-',ms=3,color=colors[ib],label=rf'$\beta={beta:g}$')
    ax[j].set(xlabel=r'$\beta m$',ylabel='Relative frequency error',title=lab[j])
ax[0].legend(frameon=False,fontsize=6,ncol=2);save(fig,'figS1')
fig,ax=plt.subplots(1,2,figsize=(7,2.6),layout='constrained');panels(ax)
r=pd.read_csv(D/'collision_refinement.csv')
for j,w in enumerate([1,2,3]):ax[0].plot(r.alpha,r[f'Tres{w}'],'o-',ms=3,color=colors[j],label=rf'$d={w}$')
ax[0].set(xlabel=r'$\alpha$',ylabel=r'$T_{\rm res}^{(d)}$',title=r'Local scan at $\Delta\Phi=\pi$');ax[0].legend(frameon=False)
z=np.load(D/'collision_residence.npz');ax[1].plot(z['t'],z['Rmol'],c=colors[1]);ax[1].axhspan(vals['Rmol']-2,vals['Rmol']+2,color='gray',alpha=.18)
tmin=z['t'][np.argmin(z['Rmol'])];ax[1].set(xlim=(tmin-600,tmin+750),ylim=(10,20),xlabel='$t$',ylabel=r'$R_{\rm mol}(t)$',title='Resolved approach and rebound');save(fig,'figS4')
fig,ax=plt.subplots(1,2,figsize=(7,2.7),layout='constrained');panels(ax)
scan=pd.read_csv(D/'collision_scan.csv');grid=scan.pivot(index='phase',columns='alpha',values='Tres2')
im=ax[0].imshow(grid.values,origin='lower',aspect='auto',extent=[.00425,.01025,-.0625,1.0625],cmap='viridis');ax[0].set(xlabel=r'$\alpha$',ylabel=r'$\Delta\Phi/\pi$',title='Exploratory finite-chain survey');fig.colorbar(im,ax=ax[0],label=r'$T_{\rm res}^{(2)}$',fraction=.045,pad=.03)
for tag,col,label in [('grid_p0_a6',colors[0],'$L=200$, baseline'),('coarse_max_check',colors[1],'$L=400$, tight'),('inphase_tight_800',colors[2],'$L=800$, tight')]:
    z=np.load(D/f'collision_{tag}.npz');ax[1].plot(z['t'],z['Rmol'],color=col,label=label)
ax[1].axhspan(vals['Rmol']-2,vals['Rmol']+2,color='gray',alpha=.15);ax[1].set(xlabel='$t$',ylabel=r'$R_{\rm mol}$',ylim=(0,100),title='Numerical sensitivity');ax[1].legend(frameon=False,fontsize=6)
save(fig,'figS5')
fig,ax=plt.subplots(1,3,figsize=(7,2.65),layout='constrained');panels(ax)
sr=pd.read_csv(D/'spectral_sum_rule.csv')
for j,eta in enumerate([1.25,2.,4.]):
    for ib,beta in enumerate(sorted(sr.beta.unique())):
        r=sr[(sr.beta==beta)&(sr.eta==eta)].sort_values('m')
        ax[j].semilogy(beta*r.m,r.sum_residual_absolute,'o-',ms=3,color=colors[ib],label=rf'$\beta={beta:g}$')
    ax[j].set(xlabel=r'$\beta m$',ylabel=r'$|\mathcal{S}|$',title=rf'$\eta={eta:g}$')
ax[0].legend(frameon=False,fontsize=6,ncol=2);save(fig,'figS2')

fig,ax=plt.subplots(1,3,figsize=(7,2.65),layout='constrained');panels(ax)
h=pd.read_csv(D/'hamiltonian_hopf.csv');h0=pd.read_csv(D/'HH_convergence.csv').iloc[1].epsilon_HH
for k,col in [(0,colors[0]),(1,colors[1])]:ax[0].plot((h.epsilon-h0)*1e4,h[f'imag_{k}'],c=col)
ax[0].axvline(0,c='black',ls=':',lw=.8)
ax[0].set(xlabel=r'$10^4(\varepsilon-\varepsilon_{\rm HH})$',ylabel=r'$\mathrm{Im}\,\lambda$',title='Krein collision')
rr=h[h.epsilon>h0];ax[1].plot((rr.epsilon-h0)*1e4,rr.real_1**2*1e5,'o-',ms=2.5,color=colors[1])
ax[1].set(xlabel=r'$10^4(\varepsilon-\varepsilon_{\rm HH})$',ylabel=r'$10^5(\mathrm{Re}\,\lambda)^2$',title='Quartet growth')
z=np.load(D/'dns_HH_a1e-06_tol1e-10.npz');t=z['t'];y=np.log(abs(z['c']/z['c'][0]))
ax[2].plot(t,y,color=colors[1],label='Direct integration')
ax[2].plot(t[::120],z['lambda_bdg'].real*t[::120],'o',ms=2,c='black',label='BdG prediction')
ax[2].set(xlabel='$t$',ylabel=r'$\ln|c(t)/c(0)|$',title='Modal growth');ax[2].legend(frameon=False,fontsize=6)
save(fig,'figS3')
print('FIGURES COMPLETE')
