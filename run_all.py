"""Run ARC on movie, music, and book using this Python environment."""
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    main = Path(__file__).resolve().with_name('main.py')
    if '--dataset' in sys.argv[1:]:
        raise SystemExit('run_all.py chooses all three datasets; omit --dataset')
    for dataset in ('movie', 'music', 'book'):
        print('\nRunning ARC on ' + dataset, flush=True)
        subprocess.run([sys.executable, str(main), '--dataset', dataset] + sys.argv[1:],
                       check=True)
