"""Read-only, post-timing coordinate replay and paired complete-tree audit."""
import argparse,json,gzip
from pathlib import Path
import numpy as np
import frontier_generation as f
from audit_placement import replay
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--output',required=True);p.add_argument('--reference');p.add_argument('--exclusions');p.add_argument('directories',nargs='+');a=p.parse_args()
bank=json.loads(Path(a.bank).read_text());assert len({g['rng_seed'] for g in bank['games']})==len(bank['games']);assert len({g['prefix']['state_hash'] for g in bank['games']})==len(bank['games'])
assert len({json.dumps(g['rng_state']) for g in bank['games']})==len(bank['games'])
assert all(state_hash(f.ref.e.state(g['prefix']['moves']))==g['prefix']['state_hash'] for g in bank['games'])
assert all(f.ref.e.sha(f'artifacts/ppo-gen3/seed-{s}.pt')==h for s,h in bank['models'].items())
ex=set(json.load(gzip.open(a.exclusions,'rt'))['states']) if a.exclusions else set(json.loads(Path('artifacts/gpu-generation/excluded.json').read_text())['states']);idx=np.load(f.ref.e.INDEX,mmap_mode='r');out={};divergences=[];score_differences=[]
for directory in a.directories:
 root=Path(directory);rows=[];visited=set();equal=0;paired=0;actions=0;same=0;maxerr=0.;nodes=0;equal_scores=0;equal_wdl=0;max_terminal_delta=0
 for task in bank['games']:
  name=f"{task['seed']}-{task['id']}.json.gz";d=json.load(gzip.open(root/name,'rt'));assert d['task']==task
  assert replay(d['moves'])==d['scores'];assert len(d['stats'])==len(d['moves'])-len(task['prefix']['moves'])
  assert all(x['forced_pass'] or (x['depth']==6 and x['stages']==[2,4,6] and not x['fallback']) for x in d['stats']);assert all(np.isfinite(v) for x in d['stats'] for v in x.get('root_scores',[]));rows.extend(d['stats']);s=State()
  for j,m in enumerate(d['moves']):
   if j>=len(task['prefix']['moves']):visited.add(state_hash(s))
   play(s,Move(*m) if m else None,validate=True)
  assert s.done;visited.add(state_hash(s))
  if a.reference:
   ref=json.load(gzip.open(Path(a.reference)/name,'rt'));paired+=1;equal+=d['moves']==ref['moves'];equal_scores+=d['scores']==ref['scores'];equal_wdl+=int(np.sign(d['scores'][0]-d['scores'][1])==np.sign(ref['scores'][0]-ref['scores'][1]));max_terminal_delta=max(max_terminal_delta,max(abs(u-v) for u,v in zip(d['scores'],ref['scores'])));prefix=len(task['prefix']['moves'])
   for j,(x,y) in enumerate(zip(d['stats'],ref['stats'])):
    if d['moves'][:prefix+j]!=ref['moves'][:prefix+j]:break
    actions+=1;same+=d['moves'][prefix+j]==ref['moves'][prefix+j]
    if root.name.startswith('cpu'):
     for key in ('root_scores','nodes','cutoffs','cache_hits','nn_policy_calls','nn_value_calls','depth','forced_pass'):
      assert x.get(key)==y.get(key),(root,name,j,key)
    xx=x.get('root_scores',[]);yy=y.get('root_scores',[]);assert len(xx)==len(yy)
    err=max([abs(u-v) for u,v in zip(xx,yy)] or [0]);maxerr=max(maxerr,err)
    if err>2e-5:score_differences.append(dict(directory=directory,game=name,ply=prefix+j,root_error=err,reference_scores=yy,new_scores=xx,same_action=d['moves'][prefix+j]==ref['moves'][prefix+j]))
    if d['moves'][prefix+j]!=ref['moves'][prefix+j]:divergences.append(dict(directory=directory,game=name,ply=prefix+j,root_error=err,reference_scores=yy,new_scores=xx))
 out[directory]=dict(games=len(bank['games']),moves=len(rows),nodes=sum(x['nodes'] for x in rows),nn_policy=sum(x['nn_policy_calls'] for x in rows),nn_value=sum(x['nn_value_calls'] for x in rows),cutoffs=sum(x['cutoffs'] for x in rows),depth6=sum(x['depth']==6 for x in rows if not x['forced_pass']),nonpass=sum(not x['forced_pass'] for x in rows),forced_pass=sum(x['forced_pass'] for x in rows),fallback=sum(x['fallback'] and not x['forced_pass'] for x in rows),visited=len(visited),saved_overlap=len(visited&ex),training_overlap=sum(f.ref.e.contains(idx,h) for h in visited),paired_games=paired,equal_trajectories=equal,equal_terminal_scores=equal_scores,equal_terminal_wdl=equal_wdl,max_terminal_score_delta=max_terminal_delta,same_state_moves=actions,action_agreement=same/actions if actions else None,max_same_state_root_error=maxerr)
 assert not out[directory]['saved_overlap'] and not out[directory]['training_overlap']
 print(directory,'audited',len(bank['games']),flush=True)
f.ref.write(a.output,dict(results=out,divergences=divergences,score_differences=score_differences,bank_sha=f.ref.e.sha(a.bank)))
