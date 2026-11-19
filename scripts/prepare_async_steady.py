import gzip,json,random,hashlib
from pathlib import Path
import numpy as np
import frontier_generation as f
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
R=Path('artifacts/gpu-async512');path=R/'steady-bank.json';assert not path.exists()
ex=set(json.load(gzip.open(R/'final/prior-states.json.gz','rt'))['states']);sources={}
for name in ('cpu-1','async-1','parallel-1','async-2','cpu-fast-1','cpu-2'):
 for p in sorted((R/'final'/name).glob('[0-9]*.json.gz')):
  sources[str(p)]=f.ref.e.sha(p);d=json.load(gzip.open(p,'rt'));s=State()
  for raw in d['moves']:ex.add(state_hash(s));play(s,Move(*raw) if raw else None,validate=False)
  ex.add(state_hash(s))
with gzip.open(R/'steady-excluded.json.gz','wt') as h:json.dump(dict(states=sorted(ex)),h)
idx=np.load(f.ref.e.INDEX,mmap_mode='r');games=[]
for gid in range(4096):
 rs=884100000000+gid;r=random.Random(rs);prefix=f.ref.e.prefix(r,idx,ex,8);ex.add(prefix['state_hash']);games.append(dict(seed=1751401,id=gid,rng_seed=rs,prefix=prefix,rng_state=r.getstate()))
bank=dict(games=games,models={'1751401':f.ref.e.sha('artifacts/ppo-gen3/seed-1751401.pt')});f.ref.write(path,bank)
with gzip.open(path.with_suffix('.json.gz'),'wt') as h:json.dump(bank,h)
f.ref.write(R/'steady-bank-manifest.json',dict(bank_sha=f.ref.e.sha(path),excluded_sha=f.ref.e.sha(R/'steady-excluded.json.gz'),source_game_hashes=sources,plan_sha=f.ref.e.sha(R/'STEADY.md'),source_manifest=f.ref.e.sha(R/'final/measurement-manifest.json')))
