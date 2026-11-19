import json,random,argparse,gzip
from pathlib import Path
import numpy as np
import frontier_generation as f
p=argparse.ArgumentParser();p.add_argument('--per-seed',type=int,default=1024);p.add_argument('--offset',type=int,default=941000000);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);assert not out.exists()
ex=set(json.loads(Path('artifacts/gpu-generation/excluded.json').read_text())['states'])
for path in [Path('artifacts/gpu-generation/bank.json'),Path('artifacts/gpu-frontier/bank.json')]:
 ex.update(g['prefix']['state_hash'] for g in json.loads(path.read_text())['games'])
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
sources={}
for directory in [Path('artifacts/gpu-generation/cpu-1'),Path('artifacts/gpu-generation/gpu-1'),Path('artifacts/gpu-frontier/cpu-1'),Path('artifacts/gpu-frontier/parallel-1')]+list(Path('artifacts/gpu-async512').glob('dev*'))+list(Path('artifacts/gpu-async512').glob('curve-*')):
 if not directory.is_dir():continue
 for path in directory.glob('[0-9]*-*.json.gz'):
  d=json.load(gzip.open(path,'rt'));z=State();sources[str(path)]=f.ref.e.sha(path)
  for move in d['moves']:
   ex.add(state_hash(z));play(z,Move(*move) if move else None,validate=False)
idx=np.load(f.ref.e.INDEX,mmap_mode='r');games=[]
for seed in f.ref.e.SEEDS:
 for gid in range(a.per_seed):
  rs=a.offset+seed*10000+gid;r=random.Random(rs);pr=f.ref.e.prefix(r,idx,ex,8);ex.add(pr['state_hash'])
  games.append(dict(seed=seed,id=gid,rng_seed=rs,prefix=pr,rng_state=r.getstate()))
f.ref.write(out,dict(games=games,models={str(s):f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in f.ref.e.SEEDS},purpose='throughput only; never training'))

f.ref.write(out.with_name(out.stem+'-exclusion-sources.json'),dict(sources=sources,excluded_count=len(ex),training_index=str(f.ref.e.INDEX),note='prior exclusion union plus every visited state from listed development/references; new starts tested against original training index'))
