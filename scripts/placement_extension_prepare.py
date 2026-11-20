"""Fix additional independent train/dev bank; prior completed experiment is read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import gzip,json,random,time,hashlib
from pathlib import Path
import numpy as np
import placement_experiment as e
from placement_large_prepare import collect_moves,gzwrite
R=Path('artifacts/value-placement-49152');PREV=Path('artifacts/value-placement-24576')
def main():
 e.setup();start=time.perf_counter();assert not (R/'bank-manifest.json').exists();sources={};ex=set()
 for name in ('exclusions.json.gz','train-visited.json.gz','dev-visited.json.gz'):
  p=PREV/name;ex.update(json.load(gzip.open(p,'rt'))['states']);sources[str(p)]=e.sha(p)
 for phase in ('validation','confirmation'):
  bp=PREV/f'{phase}-bank.json';prefixes=json.loads(bp.read_text())['prefixes'];sources[str(bp)]=e.sha(bp)
  ex.update(p['state_hash'] for p in prefixes)
  for p in PREV.glob(f'{phase}-??-*.json.gz'):
   sources[str(p)]=e.sha(p)
   for g in json.load(gzip.open(p,'rt'))['games']:collect_moves(prefixes[g['prefix']]['moves']+g['moves'],ex)
 gzwrite(R/'exclusions.json.gz',dict(states=sorted(ex)))
 idx=np.load(e.INDEX,mmap_mode='r');index_sha=hashlib.sha256(memoryview(idx).cast('B')).hexdigest();assert index_sha=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c'
 games=[]
 for si,seed in enumerate(e.SEEDS):
  for phase,n,base in [('train',8192,886200000000),('dev',64,886300000000)]:
   for gid in range(n):
    rs=base+si*100000+gid;rng=random.Random(rs);pr=e.prefix(rng,idx,ex,8);ex.add(pr['state_hash'])
    games.append(dict(seed=seed,id=gid if phase=='train' else 100000+gid,phase=phase,rng_seed=rs,prefix=pr,rng_state=rng.getstate()))
    if gid%1024==0:print('BANK',seed,phase,gid,flush=True)
 bank=dict(games=games,models={str(s):e.sha(f'artifacts/ppo-gen3/seed-{s}.pt') for s in e.SEEDS},purpose='new independent extension training/development only')
 e.write(R/'bank.json',bank);gzwrite(R/'bank.json.gz',bank)
 e.write(R/'bank-manifest.json',dict(games=len(games),train_games=24576,dev_games=192,bank_sha=e.sha(R/'bank.json'),compressed_sha=e.sha(R/'bank.json.gz'),excluded_sha=e.sha(R/'exclusions.json.gz'),index_array_sha=index_sha,sources=sources,plan_sha=e.sha(R/'PLAN.md'),seconds=time.perf_counter()-start))
 print('BANK FIXED',len(games),flush=True)
if __name__=='__main__':main()
