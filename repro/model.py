"""Reproducible Salerno stationary, fold, and spectral calculations.

All computations use zero exterior boundary values.  beta fixes omega;
sigma is the reflection parity of the real two-pulse profile.
"""
import numpy as np
from scipy.optimize import root, root_scalar, minimize_scalar
from scipy.linalg import eig, eigvals, svd, eigh
from scipy.optimize import linear_sum_assignment

def al(n,beta,X):
    # An overflow of cosh in the far exterior represents a zero tail in float64.
    with np.errstate(over='ignore'):
        return np.sinh(beta)/np.cosh(beta*(n-X))

def al_x(n,beta,X):
    q=al(n,beta,X)
    return beta*q*np.tanh(beta*(n-X))

def melnikov(X,beta,derivative=False):
    n=np.arange(-int(np.ceil(40/beta)),int(np.ceil(40/beta))+1)
    q=al(n,beta,X);qx=al_x(n,beta,X)
    if not derivative:return 2*np.sum(qx*q**3/(1+q*q))
    qxx=beta*beta*q*(2*np.tanh(beta*(n-X))**2-1)
    return 2*np.sum(qxx*q**3/(1+q*q)+qx*qx*q*q*(3+q*q)/(1+q*q)**2)

def fold_theory(beta,sigma):
    interval=(-.49,-.01) if sigma==1 else (.01,.49)
    x=root_scalar(lambda x:melnikov(-x,beta,True)-2*beta*melnikov(-x,beta),bracket=interval,xtol=5e-14).root
    K=8*beta*np.sinh(beta)**3
    C=-sigma*K*np.exp(-2*beta*x)/melnikov(-x,beta)
    return x,C,K

def branch_x(beta,sigma,eta,companion=False):
    xf,C,K=fold_theory(beta,sigma)
    interval=(xf,.4999999) if companion else (1e-8,xf)
    if sigma==1:interval=(-.4999999,xf) if companion else (xf,-1e-8)
    f=lambda x:eta*C*melnikov(-x,beta)+sigma*K*np.exp(-2*beta*x)
    return root_scalar(f,bracket=interval,xtol=1e-13).root

def neighbors(q):
    out=np.zeros_like(q);out[1:]+=q[:-1];out[:-1]+=q[1:];return out

def stationary_F(q,beta,epsilon):
    return -2*np.cosh(beta)*q+(1+q*q)*neighbors(q)+2*epsilon*q**3

def stationary_J(q,beta,epsilon,phase=False):
    diag=-2*np.cosh(beta)+2*epsilon*q*q if phase else -2*np.cosh(beta)+2*q*neighbors(q)+6*epsilon*q*q
    return np.diag(diag)+np.diag((1+q*q)[:-1],1)+np.diag((1+q*q)[1:],-1)

def hessian_contraction(q,v,epsilon):
    diag=2*neighbors(q)*v+2*q*neighbors(v)+12*epsilon*q*v
    return np.diag(diag)+np.diag((2*q*v)[:-1],1)+np.diag((2*q*v)[1:],-1)

class Lattice:
    def __init__(self,beta,m=0,sigma=-1,padding=32,L=None):
        self.beta=float(beta);self.m=int(m);self.sigma=int(sigma)
        self.c=.5 if m%2 else 0.
        if L is None:L=int(np.ceil(m/2+padding/beta))
        self.n=np.arange(-L,L+1+int(m%2));N=len(self.n)
        cols=[]
        for j in range((N+1)//2):
            k=N-1-j
            if j==k:
                if sigma==1:
                    col=np.zeros(N);col[j]=1;cols.append(col)
            else:
                col=np.zeros(N);col[j]=1/np.sqrt(2);col[k]=sigma/np.sqrt(2);cols.append(col)
        self.P=np.array(cols).T

    def guess(self,x):
        R=self.m+2*x
        return al(self.n,self.beta,self.c-R/2)+self.sigma*al(self.n,self.beta,self.c+R/2)

    def solve(self,epsilon,x=None,q0=None,tol=1e-11):
        if q0 is None:
            if x is None:
                xf,C,K=fold_theory(self.beta,self.sigma)
                eta=epsilon/(C*np.exp(-self.beta*self.m))
                x=branch_x(self.beta,self.sigma,max(1.0001,eta))
            q0=self.guess(x)
        P=self.P
        fun=lambda a:P.T@stationary_F(P@a,self.beta,epsilon)
        jac=lambda a:P.T@stationary_J(P@a,self.beta,epsilon)@P
        sol=root(fun,P.T@q0,jac=jac,tol=tol)
        q=P@sol.x
        residual=np.max(np.abs(stationary_F(q,self.beta,epsilon)))
        if residual>1e-10:raise RuntimeError(('stationary',self.beta,self.m,epsilon,sol.message,residual))
        return q,residual

    def fold(self,tol=1e-10):
        P=self.P;k=P.shape[1];xf,C,K=fold_theory(self.beta,self.sigma)
        q0=self.guess(xf);a0=P.T@q0;e0=C*np.exp(-self.beta*self.m)
        J0=P.T@stationary_J(q0,self.beta,e0)@P
        v0=svd(J0)[2][-1]
        def fun(z):
            a=z[:k];v=z[k:2*k];e=z[-1];q=P@a;J=P.T@stationary_J(q,self.beta,e)@P
            return np.r_[P.T@stationary_F(q,self.beta,e),J@v,np.dot(v,v)-1]
        def jac(z):
            a=z[:k];v=z[k:2*k];e=z[-1];q=P@a;vv=P@v
            J=P.T@stationary_J(q,self.beta,e)@P
            T=P.T@hessian_contraction(q,vv,e)@P
            mat=np.zeros((2*k+1,2*k+1));mat[:k,:k]=J;mat[:k,-1]=P.T@(2*q**3)
            mat[k:2*k,:k]=T;mat[k:2*k,k:2*k]=J;mat[k:2*k,-1]=P.T@(6*q*q*vv)
            mat[-1,k:2*k]=2*v
            return mat
        z0=np.r_[a0,v0,e0]
        sol=root(fun,z0,jac=jac,tol=tol)
        if max(abs(fun(sol.x)))>1e-9 or sol.x[-1]<=0:raise RuntimeError(('fold',self.beta,self.m,self.sigma,sol.message,max(abs(fun(sol.x)))))
        q=P@sol.x[:k];v=P@sol.x[k:2*k];eps=sol.x[-1]
        # WJ is symmetric; the left null vector is Wv, normalized by w^T v=1.
        w=v/(1+q*q);w/=np.dot(w,v)
        trans=np.dot(w,2*q**3);curv=np.dot(w,hessian_contraction(q,v,eps)@v)
        data=dict(beta=self.beta,sigma=self.sigma,m=self.m,epsilon=eps,prediction=e0,rho=eps/e0,
                  relative_error=abs(eps/e0-1),residual=max(abs(stationary_F(q,self.beta,eps))),
                  null_residual=max(abs(stationary_J(q,self.beta,eps)@v)),transversality=trans,quadratic=curv,
                  points=len(q),xstar=xf,C=C)
        return q,v,data

def spectrum(q,beta,epsilon,left=False):
    lp=stationary_J(q,beta,epsilon);lm=stationary_J(q,beta,epsilon,phase=True);zero=np.zeros_like(lp)
    A=np.block([[zero,-lm],[lp,zero]])
    if left:
        ev,vl,vr=eig(A,left=True,right=True)
        return ev,vr,vl,A
    ev,vr=eig(A)
    return ev,vr,A

def krein(q,beta,epsilon,w):
    N=len(q);hp=-stationary_J(q,beta,epsilon)/(1+q*q)[:,None];hm=-stationary_J(q,beta,epsilon,True)/(1+q*q)[:,None]
    return float(np.real(np.vdot(w[:N],hp@w[:N])+np.vdot(w[N:],hm@w[N:])))

def geometry(n,q,c=0.):
    nn=np.log1p(abs(q)**2);left=n<c;right=n>c
    nl=np.sum(nn[left]);nr=np.sum(nn[right]);total=np.sum(nn)
    xl=np.sum(n[left]*nn[left])/nl;xr=np.sum(n[right]*nn[right])/nr
    return dict(norm=total,H=None,XL=xl,XR=xr,Rmol=xr-xl,center=np.sum(n*nn)/total,asym=(nr-nl)/total)

def invariants(q,epsilon):
    amp=abs(q)**2
    norm=np.sum(np.log1p(amp));H=-2*np.real(np.sum(q[:-1]*np.conj(q[1:])))-2*epsilon*np.sum(amp-np.log1p(amp))
    return norm,H

def identify_modes(lat,q,epsilon):
    ev,vr,A=spectrum(q,lat.beta,epsilon);N=len(q)
    fit=minimize_scalar(lambda R:np.sum((q-(al(lat.n,lat.beta,lat.c-R/2)-al(lat.n,lat.beta,lat.c+R/2)))**2),
                        bounds=(lat.m,lat.m+1),method='bounded',options={'xatol':1e-13})
    Rfit=fit.x;ql=al(lat.n,lat.beta,lat.c-Rfit/2);qr=al(lat.n,lat.beta,lat.c+Rfit/2)
    qlx=al_x(lat.n,lat.beta,lat.c-Rfit/2);qrx=al_x(lat.n,lat.beta,lat.c+Rfit/2)
    tans=np.array([ql+qr,qlx-qrx,-.5*(qlx+qrx)])
    ids=np.where((ev.imag>1e-6)&(ev.imag<2*(np.cosh(lat.beta)-1)*.95)&(abs(ev.real)<1e-7))[0]
    overlaps=np.empty((3,len(ids)))
    for j in range(3):
        for kk,k in enumerate(ids):
            v=vr[N:,k] if j==0 else vr[:N,k]
            overlaps[j,kk]=abs(np.vdot(v,tans[j]))/(np.linalg.norm(v)*np.linalg.norm(tans[j]))
    rows,cols=linear_sum_assignment(-overlaps);indices=ids[cols]
    if len(indices)!=3:raise RuntimeError(('modes',lat.beta,lat.m,epsilon,ev[ids]))
    matrix=np.empty((3,3));modes=[]
    for j,k in enumerate(indices):
        v=vr[N:,k] if j==0 else vr[:N,k];phase=np.angle(np.vdot(tans[j],v));v=v*np.exp(-1j*phase);modes.append(v.real)
        for l in range(3):matrix[j,l]=abs(np.vdot(v,tans[l]))/(np.linalg.norm(v)*np.linalg.norm(tans[l]))
    return dict(frequencies=ev[indices].imag,indices=indices,ev=ev,vr=vr,overlaps=matrix,tangents=tans,modes=np.array(modes),
                Rfit=Rfit,krein=np.array([krein(q,lat.beta,epsilon,vr[:,k]) for k in indices]),
                eig_residuals=np.array([np.linalg.norm(A@vr[:,k]-ev[k]*vr[:,k])/np.linalg.norm(vr[:,k]) for k in indices]))

def frequency_theory(beta,m,eta):
    x=branch_x(beta,-1,eta);_,C,_=fold_theory(beta,-1);s=np.sinh(beta)
    eps=eta*C*np.exp(-beta*m);deltaR=np.exp(-beta*(m+2*x))
    exchange2=32*s**4*deltaR
    pin2=-2*s*eps*melnikov(-x,beta,True)/(beta*beta)
    return np.sqrt([exchange2,pin2,pin2+exchange2]),x

def hh_threshold(padding=32):
    lat=Lattice(1,5,-1,padding=padding)
    def discriminant(e):
        q,res=lat.solve(e,x=.1)
        mu=eigvals(-stationary_J(q,1,e,True)@stationary_J(q,1,e))
        pair=mu[np.argsort(abs(mu+.4128**2))[:2]]
        return float(np.real((pair[0]-pair[1])**2))
    return root_scalar(discriminant,bracket=(.52670,.52673),xtol=1e-13).root
