"""Fix genuinely new train/dev banks for the preregistered 24,576-game trial."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import gzip,json,random,time,hashlib
from pathlib import Path
import numpy as np
import placement_experiment as e
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
R=Path('artifacts/value-placement-24576')
def gzwrite(p,x):
 with gzip.open(p,'wt') as h:json.dump(x,h,separators=(',',':'))
def collect_moves(ms,ex):
 s=State()
 for m in ms:ex.add(state_hash(s));play(s,Move(*m) if m else None,validate=False)
 ex.add(state_hash(s))
def main():
 start=time.perf_counter();assert not (R/'bank.json').exists();ex=set();sources={}
 for base in ('value-placement-ab','value-placement-scale','value-placement-6144'):
  root=Path('artifacts')/base
  for name in ('exclusions.json','historical-starts.json','train-labels.json','dev-labels.json'):
   p=root/name
   if p.exists():
    d=json.loads(p.read_text());ex.update(d.get('states',[]));ex.update(d.get('visited',[]));sources[str(p)]=e.sha(p)
  for bp in root.rglob('*bank.json'):
   b=json.loads(bp.read_text());prefixes=b.get('prefixes',[]);ex.update(p['state_hash'] for p in prefixes);sources[str(bp)]=e.sha(bp)
   phase=bp.name.removesuffix('-bank.json')
   for gp in bp.parent.glob(f'{phase}-??-*.json.gz'):
    d=json.load(gzip.open(gp,'rt'));sources[str(gp)]=e.sha(gp)
    for g in d['games']:collect_moves(prefixes[g['prefix']]['moves']+g['moves'],ex)
 print('old learning/evaluation excluded',len(ex),flush=True)
 unique=set()
 for base in ('gpu-generation','gpu-frontier','gpu-async512'):
  for p in sorted((Path('artifacts')/base).rglob('[0-9]*-*.json.gz')):
   sources[str(p)]=e.sha(p);d=json.load(gzip.open(p,'rt'));key=json.dumps(d['moves'],separators=(',',':'))
   if key in unique:continue
   unique.add(key);collect_moves(d['moves'],ex)
 print('all prior speed games excluded',len(ex),'unique',len(unique),flush=True)
 gzwrite(R/'exclusions.json.gz',dict(states=sorted(ex)))
 idx=np.load(e.INDEX,mmap_mode='r');index_sha=hashlib.sha256(memoryview(idx).cast('B')).hexdigest();assert index_sha=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 games=[]
 for si,seed in enumerate(e.SEEDS):
  for phase,n,base in [('train',6144,885200000000),('dev',64,885300000000)]:
   for gid in range(n):
    rs=base+si*100000+gid;rng=random.Random(rs);pr=e.prefix(rng,idx,ex,8);ex.add(pr['state_hash'])
    games.append(dict(seed=seed,id=gid if phase=='train' else 100000+gid,phase=phase,rng_seed=rs,prefix=pr,rng_state=rng.getstate()))
    if gid%512==0:print('bank',seed,phase,gid,flush=True)
 bank=dict(games=games,models={str(s):e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in e.SEEDS},purpose='new training and development only, disjoint from past throughput and strength evaluation')
 e.write(R/'bank.json',bank);gzwrite(R/'bank.json.gz',bank)
 e.write(R/'bank-manifest.json',dict(games=18624,train_games=18432,dev_games=192,bank_sha=e.sha(R/'bank.json'),compressed_sha=e.sha(R/'bank.json.gz'),excluded_sha=e.sha(R/'exclusions.json.gz'),index_array_sha=index_sha,sources=sources,prior_unique_speed_games=len(unique),plan_sha=e.sha(R/'PLAN.md'),seconds=time.perf_counter()-start))
 print('BANK FIXED',len(games),time.perf_counter()-start,flush=True)
if __name__=='__main__':main()
