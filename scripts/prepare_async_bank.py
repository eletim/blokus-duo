import json,random,argparse
from pathlib import Path
import numpy as np
import frontier_generation as f
p=argparse.ArgumentParser();p.add_argument('--per-seed',type=int,default=1024);p.add_argument('--offset',type=int,default=941000000);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);assert not out.exists()
ex=set(json.loads(Path('artifacts/gpu-generation/excluded.json').read_text())['states'])
for path in [Path('artifacts/gpu-generation/bank.json'),Path('artifacts/gpu-frontier/bank.json')]:
 ex.update(g['prefix']['state_hash'] for g in json.loads(path.read_text())['games'])
idx=np.load(f.ref.e.INDEX,mmap_mode='r');games=[]
for seed in f.ref.e.SEEDS:
 for gid in range(a.per_seed):
  rs=a.offset+seed*10000+gid;r=random.Random(rs);pr=f.ref.e.prefix(r,idx,ex,8);ex.add(pr['state_hash'])
  games.append(dict(seed=seed,id=gid,rng_seed=rs,prefix=pr,rng_state=r.getstate()))
f.ref.write(out,dict(games=games,models={str(s):f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in f.ref.e.SEEDS},purpose='throughput only; never training'))
