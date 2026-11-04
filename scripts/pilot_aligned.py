import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.aligned_data import collect,quality
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/search-aligned/pilot.npz');a=p.parse_args()
    path=Path(a.output)
    collect(path,games=12,seed=290000,groups=2,repeats=12)
    print(quality(path),flush=True)
