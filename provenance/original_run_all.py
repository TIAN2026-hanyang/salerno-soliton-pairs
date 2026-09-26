"""Recompute the revised paper's numerical tables and figures in dependency order."""
import os
import subprocess
import sys
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1';env['OMP_NUM_THREADS']='1'
    commands=[['run_static.py'],['run_reduced_checks.py'],['run_dynamics.py','dns'],['run_dynamics.py','pilot'],
              ['run_dynamics.py','scan'],['run_dynamics.py','refine'],['run_checks.py'],
              ['run_collision_sensitivity.py'],['make_tables.py'],['make_figures.py']]
    for args in commands:
        print('Running:',' '.join(args),flush=True)
        subprocess.run([sys.executable,str(root/args[0]),*args[1:]],check=True,env=env,cwd=root.parent)

if __name__=='__main__':main()
