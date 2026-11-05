"""Frozen 2048 confirmation; no model selection on this bank."""
import json
from pathlib import Path
import numpy as np
from . import ppo_curve_eval as q,terminal_eval as e
from .refinement_search import RefinementAgent
from .aligned_data import BASE
from .tournament import digest,read_prefixes,prefix_state,summarize
from .visited_data import state_hash
from .engine import Move,is_legal,legal_moves,play
from .research import sha
from .value_experiment import write
R=Path('artifacts/ppo-4096');OLD=q.R

def run():
 ps,ss,sources=q.forbidden();f=e.disjoint_fixture(1570000,96,ps,ss);write(R/'confirmation-prefixes.json',f)
 models={str(s):str(OLD/f'seed-{s}/iteration-2048.pt') for s in (1401,1402,1403)}
 write(R/'confirmation-lock.json',dict(models={s:dict(path=p,sha256=sha(p)) for s,p in models.items()},teacher=dict(path=str(BASE),sha256=sha(BASE),budget_ms=100,method='priority',reserve_fraction=0),prefix_sha256=f['sha256'],training_index_sha256=sha(OLD/'training-states/index.json'),prior_sources=sources,preflight_pass=True,adopted_start_overlap=0,selection='none; default example seed1401 preregistered'))
 assert not {state_hash(prefix_state(p)) for p in f['prefixes']}&ss
 factory=e.agent;reset=e.reset_game
 e.agent=lambda spec:RefinementAgent(BASE,100,'priority','cuda') if spec=='teacher100' else factory(spec)
 def cold(ag):
  reset(ag)
  if isinstance(ag,RefinementAgent):ag.reset_game()
 e.reset_game=cold
 for s,path in models.items():
  specs=[('oldvalue','oldvalue'),('teacher100','teacher100')]
  if s=='1401':specs.append(('resume512','artifacts/terminal-ppo/extended/seed-1401/iteration-512.pt'))
  for name,spec in specs:
   result=e.match(path,spec,f)
   if name=='teacher100':result['right_sha256']=sha(BASE)
   write(R/f'confirmation-{s}-{name}.json',result);print(s,name,result['summary'],flush=True)
 e.agent=factory;e.reset_game=reset
 audit()

def audit():
 f=read_prefixes(R/'confirmation-prefixes.json');lock=json.loads((R/'confirmation-lock.json').read_text());train=q.hashes('training-states');lookup={p['id']:p for p in f['prefixes']};assert lock['prefix_sha256']==f['sha256'];assert not {state_hash(prefix_state(p)) for p in f['prefixes']}&train
 visited=set();reports={};games=0
 for path in R.glob('confirmation-1*.json'):
  r=json.loads(path.read_text());assert r['prefix_sha256']==f['sha256'] and sha(r['left'])==r['left_sha256'];assert r['summary']==summarize(r['games']);calls=0
  for g in r['games']:
   s=prefix_state(lookup[g['prefix_id']])
   for row in g['moves']:
    visited.add(state_hash(s));m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);calls+=1
   assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);games+=1
  assert calls==sum(t['calls'] for t in r['timing']);reports[path.stem]=r['summary']
 aggregate={}
 for op in ('oldvalue','teacher100'):
  arrays=[];times=[[],[]]
  for seed in (1401,1402,1403):
   r=json.loads((R/f'confirmation-{seed}-{op}.json').read_text());groups={}
   for g in r['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
   arrays.append(np.array([[np.mean([float(d>0)+.5*float(d==0) for d in ds]),np.mean(ds)] for ds in groups.values()]))
   for i in (0,1):times[i]+=r['timing'][i]['all_ms']
  x=np.mean(arrays,0);ix=np.random.default_rng(1571000).integers(len(x),size=(4000,len(x)));ci=np.quantile(x[ix].mean(1),[.025,.975],axis=0)
  aggregate[op]=dict(point=float(x[:,0].mean()),difference=float(x[:,1].mean()),point_ci95=ci[:,0].tolist(),difference_ci95=ci[:,1].tolist(),mean_ms=[float(np.mean(t)) for t in times])
 write(R/'confirmation-audit.json',dict(pass_=True,games=games,prefixes=96,train_actual_overlap=len(train&visited),aggregate=aggregate,individual=reports,model_reselection=False))
 print(aggregate,flush=True)
if __name__=='__main__':run()
