"""Replay legal reply groups and terminal labels; usable for pilot and full data."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State
from blokus.aligned_data import initialize,replies_group,finish
from blokus.value_search import encode
from blokus.research import sha
from blokus.value_experiment import write


def check(path):
    path=Path(path);d=np.load(path);meta=json.loads(path.with_suffix('.groups.json').read_text());initialize();count=0
    assert sha(path)==json.loads(path.with_suffix('.json').read_text())['sha256'] and len(meta)==len(d['game'])
    assert d['mask'].any(-1).all() and np.isfinite(d['returns']).all()
    for i in np.linspace(0,len(meta)-1,min(4,len(meta)),dtype=int):
        g=meta[i];leaves,moves,replies=replies_group(State(**g['root']))
        assert moves==g['moves'] and replies==g['replies']
        for k,branch in enumerate(leaves):
            for j,leaf in enumerate(branch):
                b,x=encode(leaf);assert np.array_equal(np.packbits(b.reshape(-1)),d['board'][i,k,j]) and np.array_equal(x,d['state'][i,k,j])
                if j==0:
                    for r in range(4):
                        value,_=finish(leaf,g['continuation_seeds'][r],r)
                        assert abs(value-float(d['returns'][i,k,j,r]))<1e-7;count+=1
    result=dict(status='PASS',groups=len(meta),label_replays=count,sha256=sha(path));write(path.with_suffix('.audit.json'),result);return result
if __name__=='__main__':print(check(sys.argv[1]))
