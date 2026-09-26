"""Recompute into a separate directory, preserving the archived results."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=['main', 'full'], default='full')
    p.add_argument('--output-dir', type=Path, default=BASE/'recomputed')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    out = args.output_dir.resolve()
    protected = [BASE/'data', BASE/'repro', BASE/'figures', BASE/'provenance', BASE/'audit']
    if out == BASE or BASE.is_relative_to(out) or any(out.is_relative_to(x) for x in protected):
        p.error('Use a separate output directory, not the archive or a protected subdirectory.')
    commands = [['run_static.py'], ['run_reduced_checks.py'], ['run_dynamics.py', 'dns']]
    if args.scope == 'full':
        commands += [['run_dynamics.py', x] for x in ['pilot', 'scan', 'refine']]
        commands += [['run_checks.py'], ['run_collision_sensitivity.py']]
    commands += [['make_pla_figures.py']]
    for command in commands:
        print('python repro/'+' '.join(command), flush=True)
    if args.dry_run:
        return
    if out.exists() and any(out.iterdir()):
        p.error('Output directory must be empty; choose a new directory for each run.')
    out.mkdir(parents=True, exist_ok=True)
    (out/'data').mkdir()
    shutil.copytree(BASE/'repro', out/'repro', ignore=shutil.ignore_patterns('__pycache__'))
    for source in BASE.glob('plot_*.py'):
        shutil.copy2(source, out/source.name)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for command in commands:
        subprocess.run([sys.executable, str(out/'repro'/command[0]), *command[1:]],
                       cwd=out, env=env, check=True)
    print('Recomputed results:', out)

if __name__ == '__main__':
    main()
