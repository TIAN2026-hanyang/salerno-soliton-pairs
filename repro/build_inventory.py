"""Build a file-by-file data inventory without modifying the numerical data."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]

def source(name):
    if name.startswith('collision_'):
        if 'inphase_' in name: return 'historical extension','repro/run_collision_sensitivity.py','In-phase domain/tolerance sensitivity'
        if name in ['collision_output_check.csv','collision_dense_output.npz','collision_coarse_max_check.npz']:
            return 'historical extension','repro/run_checks.py','Output sampling and extended domain checks'
        if name.startswith('collision_grid_') or name=='collision_scan.csv': stage='scan'
        elif name.startswith('collision_refine_') or name in ['collision_refinement.csv','collision_sensitivity.csv','collision_long_tight.npz','collision_large_domain.npz','collision_matched_large.npz']: stage='refine'
        elif name.startswith('collision_initial_'): stage='prepare_collision (called by pilot/scan/refine/sensitivity)'
        else: stage='pilot'
        return 'historical extension','repro/run_dynamics.py '+stage,'Collision initial state, trajectory or summary'
    if name=='HH_convergence.csv':return 'historical extension','repro/run_checks.py','HH domain check'
    if name.startswith('dns_'):
        scope='mixed: 7 current + 2 HH rows' if name=='dns_benchmarks.csv' else ('historical extension' if name.startswith('dns_HH') else 'current manuscript')
        return scope,'repro/run_dynamics.py dns','Local mode dynamics; summary or 2401-sample trajectory'
    if name.startswith(('softening_','spectral_sum_rule','reduced_checks_')):
        return 'current manuscript','repro/run_reduced_checks.py','Center-zero or sum-rule result/validation'
    if name in ['representative_HH.npz','hamiltonian_hopf.csv']:
        return 'historical extension','repro/run_static.py','Finite-epsilon HH data'
    scope='mixed: current and historical metadata' if name in ['representative_values.json','environment.json'] else 'current manuscript'
    return scope,'repro/run_static.py','Static state, fold, spectrum, branch or validation metadata'

def main():
    rows=[]
    for f in sorted((BASE/'data').iterdir()):
        if not f.is_file():continue
        scope,generator,purpose=source(f.name)
        row=dict(file='data/'+f.name,scope=scope,generator=generator,purpose=purpose,
                 size_bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
        if f.suffix=='.csv':
            d=pd.read_csv(f);row.update(rows=len(d),columns=list(d.columns))
        elif f.suffix=='.npz':
            with np.load(f,allow_pickle=False) as d:
                row['arrays']={k:dict(shape=list(d[k].shape),dtype=str(d[k].dtype)) for k in d.files}
        elif f.suffix=='.json':row['keys']=list(json.loads(f.read_text()))
        rows.append(row)
    (BASE/'DATA_FILE_INDEX.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print('Indexed',len(rows),'data files')

if __name__=='__main__':main()
