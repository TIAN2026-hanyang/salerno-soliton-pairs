"""V8: test center-mode softening and the leading squared-frequency sum rule.

Run after run_static.py. Existing V7 data are read without alteration. The
secondary-zero solve keeps the parent profile in its odd reflection sector
and the stationary null vector in the opposite (even) sector.
"""
from pathlib import Path
import io
import json
import os
import platform
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
from scipy.optimize import root, root_scalar
from model import (Lattice, al_x, fold_theory, melnikov, stationary_F,
                   stationary_J, hessian_contraction, geometry)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'


def atomic(path, data):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_csv(name, rows):
    atomic(DATA / name, pd.DataFrame(rows).to_csv(index=False).encode())


def softening_theory(beta):
    xf, cf, k = fold_theory(beta, -1)
    x = root_scalar(lambda x: melnikov(-x, beta, True),
                    bracket=(1e-6, xf), xtol=5e-14).root
    c = k * np.exp(-2 * beta * x) / melnikov(-x, beta)
    return x, c, c / cf


def secondary_zero(beta, m, padding=32):
    lat = Lattice(beta, m, -1, padding=padding)
    odd = lat.P
    even = Lattice(beta, m, 1, padding=padding).P
    ko, ke = odd.shape[1], even.shape[1]
    xs, cs, eta_s = softening_theory(beta)
    e0 = cs * np.exp(-beta * m)
    q0 = lat.guess(xs)
    d0 = np.sqrt(1 + q0*q0)
    s0 = -stationary_J(q0, beta, e0) / d0[:, None] * d0[None, :]
    va, ve = eigh(even.T @ s0 @ even)
    v0 = even @ ve[:, np.argmin(abs(va))] * d0
    v0 /= np.linalg.norm(v0)

    def fun(z):
        q, v, e = odd @ z[:ko], even @ z[ko:ko+ke], z[-1]
        return np.r_[odd.T @ stationary_F(q, beta, e),
                     even.T @ (stationary_J(q, beta, e) @ v),
                     np.dot(v, v)-1]

    def jac(z):
        q, v, e = odd @ z[:ko], even @ z[ko:ko+ke], z[-1]
        j = stationary_J(q, beta, e)
        out = np.zeros((ko+ke+1, ko+ke+1))
        out[:ko, :ko] = odd.T @ j @ odd
        out[:ko, -1] = odd.T @ (2*q**3)
        out[ko:ko+ke, :ko] = even.T @ hessian_contraction(q, v, e) @ odd
        out[ko:ko+ke, ko:ko+ke] = even.T @ j @ even
        out[ko:ko+ke, -1] = even.T @ (6*q*q*v)
        out[-1, ko:ko+ke] = 2 * (even.T @ v)
        return out

    sol = root(fun, np.r_[odd.T @ q0, even.T @ v0, e0], jac=jac, tol=1e-11)
    q, v, e = odd @ sol.x[:ko], even @ sol.x[ko:ko+ke], sol.x[-1]
    residual = float(np.max(abs(stationary_F(q, beta, e))))
    null_residual = float(np.max(abs(stationary_J(q, beta, e) @ v)))
    if max(residual, null_residual, abs(np.dot(v, v)-1)) > 2e-11 or e <= 0:
        raise RuntimeError((beta, m, sol.message, residual, null_residual))
    # Track the same stationary eigenvector on either side by overlap, rather
    # than mistaking another near-zero eigenvalue for the selected crossing.
    def curvature(ee):
        qq, _ = lat.solve(ee, q0=q, tol=1e-11)
        dd = np.sqrt(1 + qq*qq)
        ss = -stationary_J(qq, beta, ee) / dd[:, None] * dd[None, :]
        vv, ww = eigh(even.T @ ss @ even)
        ref = even.T @ (v/dd)
        idx = np.argmax(abs(ww.T @ ref))
        return float(vv[idx])
    below, above = curvature(e*(1-1e-3)), curvature(e*(1+1e-3))
    if not below < 0 < above:
        raise RuntimeError(('No softening sign change', beta, m, below, above))
    r = m + 2*xs
    tangent = al_x(lat.n, beta, lat.c-r/2)-al_x(lat.n, beta, lat.c+r/2)
    overlap = abs(np.dot(v, tangent))/(np.linalg.norm(v)*np.linalg.norm(tangent))
    row = dict(beta=beta, m=m, padding=padding, points=len(q), epsilon_CS=e,
               prediction=e0, relative_error=abs(e/e0-1), x_CS_lead=xs,
               C_CS=cs, eta_CS_lead=eta_s, residual=residual,
               null_residual=null_residual, even_parity_residual=np.linalg.norm(v-v[::-1]),
               center_tangent_overlap=overlap, curvature_below=below,
               curvature_above=above, Rmol=geometry(lat.n, q, lat.c)['Rmol'])
    return lat.n, q, v, row


def main():
    folds = pd.read_csv(DATA/'folds_80.csv')
    opposite = folds[folds.sigma == -1].sort_values(['beta', 'm'])
    rows = []
    for f in opposite.itertuples():
        n, q, v, row = secondary_zero(f.beta, int(f.m))
        row.update(epsilon_fold_numerical=f.epsilon,
                   eta_CS_numerical=row['epsilon_CS']/f.epsilon,
                   numerical_gap=row['epsilon_CS']/f.epsilon-1)
        rows.append(row)
        buf = io.BytesIO()
        np.savez_compressed(buf, n=n, q=q, v=v, **row)
        atomic(DATA/f'softening_b{f.beta:g}_m{f.m}.npz', buf.getvalue())
        print(f'Softening beta={f.beta:g} m={f.m}: relative error {row["relative_error"]:.3g}', flush=True)
    write_csv('softening_thresholds.csv', rows)
    checks = []
    for beta, m in [(.6, 31), (1., 10), (1.5, 13)]:
        for padding in [24, 32, 40]:
            _, _, _, row = secondary_zero(beta, m, padding)
            checks.append(row)
    write_csv('softening_convergence.csv', checks)

    spec = pd.read_csv(DATA/'spectral_grid.csv')
    spec = spec.merge(opposite[['beta', 'm', 'epsilon', 'rho']].rename(
        columns={'epsilon': 'epsilon_fold_numerical', 'rho': 'fold_ratio'}),
        on=['beta', 'm'], validate='many_to_one')
    spec['eta_numerical'] = spec.epsilon / spec.epsilon_fold_numerical
    spec['sum_residual_signed'] = (spec.stretch**2-spec.pinning**2-spec.exchange**2)/spec.stretch**2
    spec['sum_residual_absolute'] = abs(spec.sum_residual_signed)
    write_csv('spectral_sum_rule.csv', spec)
    summary = []
    for (beta, eta), group in spec.groupby(['beta', 'eta']):
        group = group.sort_values('m')
        first, last = group.iloc[0], group.iloc[-1]
        summary.append(dict(beta=beta, eta=eta, m_first=int(first.m), m_last=int(last.m),
                            eta_num_first=first.eta_numerical, eta_num_last=last.eta_numerical,
                            residual_first=first.sum_residual_signed, residual_last=last.sum_residual_signed,
                            max_absolute=group.sum_residual_absolute.max()))
    write_csv('spectral_sum_rule_summary.csv', summary)
    sb = pd.DataFrame(rows)
    last_sb = sb.loc[sb.groupby('beta').m.idxmax()]
    last_sum = pd.DataFrame(summary)
    report = dict(softening_count=len(rows), spectral_state_count=len(spec),
                  softening_max_relative_error=sb.relative_error.max(),
                  softening_last_max_relative_error=last_sb.relative_error.max(),
                  softening_max_stationary_residual=sb.residual.max(),
                  softening_max_null_residual=sb.null_residual.max(),
                  minimum_center_tangent_overlap=sb.center_tangent_overlap.min(),
                  sum_rule_max_abs=spec.sum_residual_absolute.max(),
                  sum_rule_last_max_abs=abs(last_sum.residual_last).max(),
                  eta_numerical_min=spec.eta_numerical.min(), eta_numerical_max=spec.eta_numerical.max(),
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    atomic(DATA/'reduced_checks_summary.json', json.dumps(report, indent=2).encode())
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
