"""Frozen generation-2 confirmation: serialized timing, then replay audit."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
from . import ppo_curve_eval as q,terminal_eval as e
from .refinement_search import RefinementAgent
from .aligned_data import BASE
from .tournament import read_prefixes,prefix_state,summarize
from .visited_data import state_hash
from .engine import Move,is_legal,legal_moves,play
from .research import sha
from .value_experiment import write
R=Path('artifacts/ppo-generation2');TRAIN=Path('artifacts/ppo-4096');SEEDS=(1401,1402,1403)

def prepare():
 if (R/'lock.json').exists():raise FileExistsError('freeze already exists')
 q.R=TRAIN;ps,ss,sources=q.forbidden();f=e.disjoint_fixture(1620000,96,ps,ss)
 models=json.loads((TRAIN/'selection.json').read_text())
 for c in models.values():assert sha(c['path'])==c['sha256']
 starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert len(starts)==96 and not starts&ss
 previous={str(s):dict(path=f'artifacts/ppo-learning-curve/seed-{s}/iteration-2048.pt') for s in SEEDS}
 for c in previous.values():c['sha256']=sha(c['path'])
 write(R/'prefixes.json',f);write(R/'lock.json',dict(models=models,previous=previous,teacher=dict(path=str(BASE),sha256=sha(BASE),budget_ms=100,method='priority',reserve_fraction=0),training_index_sha256=sha(TRAIN/'training-states/index.json'),prefix_sha256=f['sha256'],sources=sources,forbidden_states=len(ss),forbidden_prefixes=len(ps),preflight_pass=True,adopted_overlap=0,seed_reselection=False))

def run():
 lock=json.loads((R/'lock.json').read_text());f=read_prefixes(R/'prefixes.json');assert f['sha256']==lock['prefix_sha256']
 factory=e.agent;reset=e.reset_game
 e.agent=lambda spec:RefinementAgent(BASE,100,'priority','cuda') if spec=='teacher100' else factory(spec)
 def cold(ag):
  reset(ag)
  if isinstance(ag,RefinementAgent):ag.reset_game()
 e.reset_game=cold
 try:
  for seed in SEEDS:
   c=lock['models'][str(seed)];assert sha(c['path'])==c['sha256']
   for op in ('previous','oldvalue','teacher100'):
    out=R/f'{seed}-{op}.json'
    if out.exists():raise FileExistsError(out)
    spec=lock['previous'][str(seed)]['path'] if op=='previous' else op
    if op=='previous':assert sha(spec)==lock['previous'][str(seed)]['sha256']
    else:assert sha(BASE)==lock['teacher']['sha256']
    gc.collect();r=e.match(c['path'],spec,f)
    if op=='teacher100':r['right_sha256']=sha(BASE)
    r['cold_per_game']=True;write(out,r);print(seed,op,r['summary'],flush=True)
 finally:e.agent=factory;e.reset_game=reset
 write(R/'timing-complete.json',dict(time=time.time()))

def clustered(arr):
 ix=np.random.default_rng(1621000).integers(len(arr),size=(4000,len(arr)));ci=np.quantile(arr[ix].mean(1),[.025,.975],axis=0)
 return dict(point=float(arr[:,0].mean()),difference=float(arr[:,1].mean()),point_ci95=ci[:,0].tolist(),difference_ci95=ci[:,1].tolist())

def audit():
 q.R=TRAIN;train=q.hashes('training-states');f=read_prefixes(R/'prefixes.json');lock=json.loads((R/'lock.json').read_text());lookup={p['id']:p for p in f['prefixes']}
 assert sha(TRAIN/'training-states/index.json')==lock['training_index_sha256'];assert lock['prefix_sha256']==f['sha256']
 starts={state_hash(prefix_state(p)) for p in f['prefixes']};ids=set(lookup)
 assert not starts&train
 for source in lock['sources']:
  assert sha(source['path'])==source['sha256']
  old=read_prefixes(source['path']);assert not ids&{p['id'] for p in old['prefixes']}
  assert not starts&{state_hash(prefix_state(p)) for p in old['prefixes']}
 visited=set();individual={};aggregate={};total=0
 for op in ('previous','oldvalue','teacher100'):
  arrays=[];times=[[],[]]
  for seed in SEEDS:
   r=json.loads((R/f'{seed}-{op}.json').read_text());c=lock['models'][str(seed)]
   assert r['left_sha256']==c['sha256']==sha(c['path']);assert r['prefix_sha256']==f['sha256'];assert r['summary']==summarize(r['games']);assert r['cold_per_game']
   right=lock['previous'][str(seed)] if op=='previous' else lock['teacher'];assert r['right_sha256']==right['sha256']==sha(right['path'])
   groups={};seats={};calls=0
   for g in r['games']:
    s=prefix_state(lookup[g['prefix_id']]);groups.setdefault(g['prefix_id'],[]).append(g['difference']);seats.setdefault(g['prefix_id'],[]).append(g['seat'])
    for row in g['moves']:
     visited.add(state_hash(s));m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);calls+=1
    visited.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);total+=1
   assert len(groups)==96 and all(sorted(v)==[0,1] for v in seats.values());assert calls==sum(t['calls'] for t in r['timing'])
   arr=np.array([[np.mean([float(d>0)+.5*float(d==0) for d in groups[p['id']]]),np.mean(groups[p['id']])] for p in f['prefixes']]);arrays.append(arr)
   individual[f'{seed}-{op}']=dict(summary=r['summary'],clustered=clustered(arr),timing=r['timing'])
   for i in (0,1):
    ts=r['timing'][i]['all_ms'];assert all(np.isfinite(t) and t>=0 for t in ts);times[i]+=ts
    individual[f'{seed}-{op}']['timing'][i]={k:v for k,v in r['timing'][i].items() if k!='all_ms'}
  v=clustered(np.mean(arrays,axis=0));v['per_seed_mean']=np.mean(arrays,axis=1).tolist();v['timing']=[]
  for i,ts in enumerate(times):
   budget=None if i==0 or op=='previous' else 6 if op=='oldvalue' else 100
   v['timing'].append(dict(calls=len(ts),mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=max(ts),budget_ms=budget,over_budget_rate=None if budget is None else float(np.mean(np.array(ts)>budget))))
  aggregate[op]=v
 overlap=len(visited&train);assert overlap==0,f'actual training overlap {overlap}; do not replace prefixes'
 prev=aggregate['previous'];regression=prev['difference_ci95'][1]<0 or prev['point_ci95'][1]<.5
 write(R/'audit.json',dict(pass_=True,games=total,prefixes=96,unique_visited_states=len(visited),training_actual_overlap=overlap,individual=individual,aggregate=aggregate,clear_regression=regression,default_generation='2048' if regression else 'gen2',note='Fixed models; seats and seeds clustered inside prefix. No model selection. Initial empty prefix traversal is not an evaluated decision.'))
 print(aggregate,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run','audit']);p.add_argument('--root',type=Path,default=R);a=p.parse_args();R=a.root;R.mkdir(parents=True,exist_ok=True);globals()[a.mode]()
