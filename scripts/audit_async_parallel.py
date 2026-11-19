import argparse,json,subprocess,sys,gzip,hashlib
import numpy as np
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
from pathlib import Path
import frontier_generation as f
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();r=Path(a.root);bank=json.loads((r/'bank.json').read_text());procs=[]
assert len({(g['seed'],g['id']) for g in bank['games']})==len(bank['games'])
assert len({g['rng_seed'] for g in bank['games']})==len(bank['games'])
assert len({g['prefix']['state_hash'] for g in bank['games']})==len(bank['games'])
assert len({json.dumps(g['rng_state']) for g in bank['games']})==len(bank['games'])
idx=np.load(f.ref.e.INDEX,mmap_mode='r');index_sha=hashlib.sha256(memoryview(idx).cast('B')).hexdigest();assert index_sha=='cdd1f0f1ac7a0e722a670fb541ce904187949c6982cd03fda1fa7acad30f7b4c';del idx
ex=set(json.loads(Path('artifacts/gpu-generation/excluded.json').read_text())['states']);seen_games=set()
for path,expected in json.loads((r/'bank-exclusion-sources.json').read_text())['sources'].items():
 assert f.ref.e.sha(path)==expected,path
 d=json.load(gzip.open(path,'rt'));key=tuple(tuple(m) if m else None for m in d['moves'])
 if key in seen_games:continue
 seen_games.add(key);z=State()
 for m in d['moves']:
  ex.add(state_hash(z));play(z,Move(*m) if m else None,validate=False)
 ex.add(state_hash(z))
for bp in (Path('artifacts/gpu-async512/development-bank.json'),Path('artifacts/gpu-generation/bank.json'),Path('artifacts/gpu-frontier/bank.json')):
 ex.update(g['prefix']['state_hash'] for g in json.loads(bp.read_text())['games'])
assert all(g['prefix']['state_hash'] not in ex for g in bank['games'])
prior=r/'prior-states.json.gz'
with gzip.open(prior,'wt') as handle:json.dump(dict(states=sorted(ex)),handle)
f.ref.write(r/'separation-manifest.json',dict(index_array_sha256=index_sha,excluded_states=len(ex),unique_source_games=len(seen_games),derived_prior_sha=f.ref.e.sha(prior),bank_sha=f.ref.e.sha(r/'bank.json')))
dirs=[str(r/x) for x in ('cpu-1','async-1','parallel-1','async-2','cpu-fast-1','cpu-2')]
for i in range(a.workers):
 b=dict(bank);b['games']=bank['games'][i::a.workers];bp=r/f'audit-bank-{i}.json';f.ref.write(bp,b)
 log=open(r/f'audit-{i}.log','w');cmd=[sys.executable,'scripts/audit_async.py','--bank',str(bp),'--output',str(r/f'audit-{i}.json'),'--reference',str(r/'cpu-1'),'--exclusions',str(prior)]+dirs
 procs.append((subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT),log))
for proc,log in procs:assert proc.wait()==0;log.close()
out={};div=[];score_differences=[]
for i in range(a.workers):
 d=json.loads((r/f'audit-{i}.json').read_text());div+=d['divergences'];score_differences+=d['score_differences']
 for name,row in d['results'].items():
  z=out.setdefault(name,{})
  for k,v in row.items():
   if k=='action_agreement':continue
   z[k]=max(z.get(k,0),v) if k.startswith('max_') else z.get(k,0)+v
  z['agree_count']=z.get('agree_count',0)+round(row['action_agreement']*row['same_state_moves'])
for z in out.values():z['action_agreement']=z['agree_count']/z['same_state_moves']
f.ref.write(r/'audit.json',dict(results=out,divergences=div,score_differences=score_differences,bank_sha=f.ref.e.sha(r/'bank.json'),visited_note='sum of unique states within disjoint audit shards; overlap checks examine every state'))
