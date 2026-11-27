"""Read-only summaries of completed fixed-benchmark generations."""
from pathlib import Path
from functools import lru_cache
import json,gzip,subprocess

def read(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)
R=Path('artifacts/komi4-trend')

@lru_cache(maxsize=16)
def benchmark(g):
    assert 5<=g<=15
    return json.loads(subprocess.check_output(['/home/eletim/blokus-duo-ai/.venv/bin/python','-m','blokus.trend_console',str(g)],text=True,timeout=60))

def compute_benchmark(g):
    from scripts.summarize_komi4_scale import row,summarize
    games=[read(p) for p in sorted((R/'games').glob(f'benchmark-g{g}-*.json.gz'))]
    assert len(games)==1536
    rows=[row(d) for d in games];out={}
    for opp in ['gen4','prior']:
        subset=[r for r in rows if r['opponent']==opp]
        out[opp]=dict(overall=summarize(subset),seats={str(s):summarize([r for r in subset if r['seat']==s]) for s in [0,1]},seeds={str(s):summarize([r for r in subset if r['seed']==s]) for s in [1751401,1751402,1751403]},seed_seats={str(seat):{str(s):summarize([r for r in subset if r['seed']==s and r['seat']==seat]) for s in [1751401,1751402,1751403]} for seat in [0,1]})
    used=sum(read(R/f'features/train-g{j}.json')['unique_games'] for j in range(6,g+1))
    return dict(generation=g,cumulative_unique_train_games=11000+used,opponents=out)

def data():
    out={n:read(R/(n+'.json')) if (R/(n+'.json')).exists() else None for n in ['plan','status','learning-progress','selection','prediction-test','prediction-intervals','audit','results']}
    files=list((R/'games').glob('*.json.gz'))
    out['generated']={kind:sum(p.name.startswith(kind+'-') for p in files) for kind in ['train','validation','test','benchmark','final','standard']}
    out['benchmarks']=[benchmark(g) for g in range(5,16) if sum(p.name.startswith(f'benchmark-g{g}-') for p in files)==1536]
    out['epochs']=[read(p) for p in sorted((R/'training').glob('*.json'))]
    return out

def games(group):
    allowed={f'benchmark-g{g}' for g in range(5,16)}|{f'train-g{g}' for g in range(6,16)}|{'final','standard'}
    if group not in allowed:return []
    return sorted((R/'games').glob(group+'-*.json.gz'))

if __name__=='__main__':
    import sys
    print(json.dumps(compute_benchmark(int(sys.argv[1]))))
