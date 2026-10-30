"""run the tests and check every saved result against the code.

usage: python run_checks.py   (about 3 minutes)
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKS = [('Unit tests', ['-m', 'unittest', 'discover', '-s', 'tests']),
          ('Study results', ['scripts/verify_full_study.py']),
          ('Summary tables', ['scripts/verify_study_summaries.py'])]


def main():
    passed = {}
    for name, args in CHECKS:
        started = time.time()
        print(f'>> {name}', flush=True)
        passed[name] = subprocess.run([sys.executable, *args], cwd=ROOT).returncode == 0
        print(f"{'PASS' if passed[name] else 'FAIL'}  {name} ({time.time() - started:.0f} s)\n", flush=True)
    for name, ok in passed.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not all(passed.values()):
        sys.exit(1)
    print('All checks passed.')


if __name__ == '__main__':
    main()
