"""run the whole project from the raw data to the figures, then the checks.

steps: pilot, study for seeds 100-109, summary tables, notebooks, checks.
finished steps are skipped unless --fresh is used.

usage:
  python run_pipeline.py              run what is missing, then check
  python run_pipeline.py --fresh      start from an empty results folder (about 1 hour)
  python run_pipeline.py --workers 3  run 3 seeds at a time
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEEDS = range(100, 110)
NOTEBOOKS = ('data_preparation', 'algorithm_checks', 'results', 'appendix')


def run(*args):
    print('>>', ' '.join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def complete(path):
    return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('status') == 'complete'


def execute_notebook(name):
    import nbformat
    from nbclient import NotebookClient
    path = ROOT/'notebooks'/f'{name}.ipynb'
    print('>> executing', path.relative_to(ROOT).as_posix(), flush=True)
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=None, kernel_name='python3',
                   resources={'metadata': {'path': str(path.parent)}}).execute()
    nbformat.write(notebook, path)


def main():
    parser = argparse.ArgumentParser(description='Run the Data Shapley project end to end.')
    parser.add_argument('--fresh', action='store_true', help='delete results/ and recompute everything')
    parser.add_argument('--workers', type=int, default=1, help='study seeds run in parallel')
    options = parser.parse_args()
    started = time.time()
    if options.fresh:
        shutil.rmtree(ROOT/'results', ignore_errors=True)
    # remove unfinished outputs before rerunning
    if not complete(ROOT/'results/pilot/pilot_seed100.json'):
        shutil.rmtree(ROOT/'results/pilot', ignore_errors=True)
        run('-m', 'data_shapley.pilot')
    missing = [s for s in SEEDS if not complete(ROOT/f'results/study/seed{s}/manifest.json')]
    for s in missing:
        shutil.rmtree(ROOT/f'results/study/seed{s}', ignore_errors=True)
    with ThreadPoolExecutor(max(1, options.workers)) as pool:
        list(pool.map(lambda s: run('-m', 'data_shapley.study', s), missing))
    run('-m', 'data_shapley.study_summary')
    for name in NOTEBOOKS:
        execute_notebook(name)
    print(f'Pipeline finished in {(time.time() - started)/60:.1f} minutes.\n', flush=True)
    run('run_checks.py')


if __name__ == '__main__':
    main()
