"""Render the five V10 figures from saved data; no numerical solver is called."""
import argparse
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=BASE/'data')
    p.add_argument('--output-dir', type=Path, default=BASE/'reproduced_figures')
    p.add_argument('--formats', default='pdf,png')
    a = p.parse_args()
    for number in ['01', '02', '03', '04', 'C1']:
        subprocess.run([sys.executable, str(BASE/f'plot_figure_{number}.py'),
                        '--data-dir', str(a.data_dir.resolve()), '--output-dir', str(a.output_dir.resolve()),
                        '--formats', a.formats], check=True)

if __name__ == '__main__':
    main()
