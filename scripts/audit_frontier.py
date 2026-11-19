"""Independent coordinate replay and paired fixed-work audit, after timings only."""
import gzip,json
from pathlib import Path
import numpy as np
import frontier_generation as f
from audit_placement import replay
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
R=f.R
bank=json.loads((R/'bank.json').read_text());assert len(bank['games'])==192
assert len({x['rng_seed'] for x in bank['games']})==192
assert len({x['prefix']['state_hash'] for x in bank['games']})==192
ex=set(json.loads((f.ref.R/'excluded.json').read_text())['states']);idx=np.load(f.ref.e.INDEX,mmap_mode='r')
for p in (f.ref.R/'cpu-1').glob('*.json.gz'):
 s=State()
 for m in json.load(gzip.open(p,'rt'))['moves']:
  ex.add(state_hash(s));play(s,Move(*m) if m else None)
assert all(f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
results={};pairs=0;equal=0;maxerr=0.;move_pairs=0;move_equal=0;divergences=[]
for mode,rep in [('cpu',1),('parallel',1),('parallel',2),('cpu',2)]:
 stats=[];seen=set()
 for task in bank['games']:
  name=f"{task['seed']}-{task['id']}.json.gz";d=json.load(gzip.open(R/f'{mode}-{rep}'/name,'rt'));assert d['task']==task
  assert replay(d['moves'])==d['scores'];s=State()
  for j,m in enumerate(d['moves']):
   if j>=8:seen.add(state_hash(s))
   play(s,Move(*m) if m else None,validate=True)
  assert s.done and len(d['stats'])==len(d['moves'])-8
  assert all(x['forced_pass'] or x['depth']==6 for x in d['stats']);stats.extend(d['stats'])
  if mode=='parallel':
   c=json.load(gzip.open(R/f'cpu-{rep}'/name,'rt'));pairs+=1;equal+=c['moves']==d['moves']
   for j,(a,b) in enumerate(zip(d['stats'],c['stats'])):
    if d['moves'][:j+8]!=c['moves'][:j+8]:break
    move_pairs+=1;move_equal+=d['moves'][j+8]==c['moves'][j+8]
    av=a.get('root_scores',[]);bv=b.get('root_scores',[]);assert len(av)==len(bv)
    err=max([abs(x-y) for x,y in zip(av,bv)] or [0]);maxerr=max(maxerr,err)
    if d['moves'][j+8]!=c['moves'][j+8]:divergences.append(dict(game=name,repeat=rep,ply=j+8,root_error=err,gpu_scores=av,cpu_scores=bv,cpu_move=c['moves'][j+8],gpu_move=d['moves'][j+8]))
 timing=json.loads((R/f'{mode}-{rep}.json').read_text());ms=[x['elapsed_ms'] for x in stats];nonpass=[x for x in stats if not x['forced_pass']]
 results[f'{mode}-{rep}']=dict(seconds=timing['seconds'],initialization_seconds=timing['initialization_seconds'],games_per_second=192/timing['seconds'],nodes=sum(x['nodes'] for x in stats),nn_policy=sum(x['nn_policy_calls'] for x in stats),nn_value=sum(x['nn_value_calls'] for x in stats),nonpass=len(nonpass),depth6=sum(x['depth']==6 for x in nonpass),moves=len(stats),elapsed_ms_quantiles=np.quantile(ms,[.5,.95,.99,1]).tolist(),visited=len(seen),saved_overlap=len(seen&ex),training_overlap=sum(f.ref.e.contains(idx,h) for h in seen))
ratios=[(results[f'cpu-{i}']['seconds']+results[f'cpu-{i}']['initialization_seconds'])/(results[f'parallel-{i}']['seconds']+results[f'parallel-{i}']['initialization_seconds']) for i in (1,2)]
quality=maxerr<=2e-5 and move_equal/move_pairs>=.99 and all(x['saved_overlap']==x['training_overlap']==0 for x in results.values())
out=dict(results=results,ratios_including_init=ratios,ratios_excluding_init=[results[f'cpu-{i}']['seconds']/results[f'parallel-{i}']['seconds'] for i in (1,2)],games_equal=equal,paired_games=pairs,same_state_moves=move_pairs,move_agreement=move_equal/move_pairs,max_same_state_root_error=maxerr,divergences=divergences,quality=quality,gate=quality and all(x>=1.15 for x in ratios),bank_sha=f.ref.e.sha(R/'bank.json'))
f.ref.write(R/'audit.json',out);print(json.dumps(out,indent=2))
