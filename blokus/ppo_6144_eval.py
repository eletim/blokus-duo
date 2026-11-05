"""Preregistered finite PPO continuation and frozen-generation comparisons."""
import argparse,contextlib,json,time
from pathlib import Path
import numpy as np
from . import ppo_curve_eval as q,terminal_eval as e
from .ppo_generation_eval import clustered
from .refinement_search import RefinementAgent
from .engine import Move,State,play,is_legal,legal_moves
from .tournament import digest,read_prefixes,prefix_state,summarize
from .research import sha
from .visited_data import state_hash
from .value_experiment import write
R=Path('artifacts/ppo-6144');BASE=Path('artifacts/ppo-4096');SEEDS=(1401,1402,1403)

def prepare():
 if (R/'lock.json').exists():raise FileExistsError('already frozen')
 rows={};sources=[]
 for path in sorted(Path('artifacts').rglob('*prefix*.json')):
  try:
   f=json.loads(path.read_text())
   if not isinstance(f,dict) or not isinstance(f.get('prefixes'),list):continue
   for p in f['prefixes']:
    prefix_state(p);assert p['id']==digest(p['moves']);rows[p['id']]=p
   sources.append(dict(path=str(path),sha256=sha(path),count=len(f['prefixes'])))
  except (KeyError,TypeError,ValueError):continue
 prefixes=list(rows.values());fixture=dict(format=1,count=len(prefixes),sha256=digest(prefixes),prefixes=prefixes)
 write(R/'forbidden-prefixes.json',fixture)
 gen2=json.loads((BASE/'selection.json').read_text());latest={}
 for seed in SEEDS:
  c=gen2[str(seed)];assert sha(c['path'])==c['sha256'];p=BASE/f'seed-{seed}/iteration-4096.pt';latest[str(seed)]=dict(path=str(p),sha256=sha(p),iteration=4096)
 write(R/'lock.json',dict(gen2=gen2,latest=latest,teacher=dict(path=str(e.BASE),sha256=sha(e.BASE)),forbidden_prefix_sha256=fixture['sha256'],sources=sources,unique_forbidden_starts=len({state_hash(prefix_state(p)) for p in prefixes}),criterion_weights=[.5,.25,.25],validation_seed=1650000,holdout_seed=1660000))
 print('Frozen prior prefixes',len(prefixes),flush=True)

@contextlib.contextmanager
def cold_factory():
 factory=e.agent;reset=e.reset_game
 e.agent=lambda spec:RefinementAgent(e.BASE,100,'priority','cuda') if spec=='teacher100' else factory(spec)
 def cold(ag):
  reset(ag)
  if isinstance(ag,RefinementAgent):ag.reset_game()
 e.reset_game=cold
 try:yield
 finally:e.agent=factory;e.reset_game=reset

def lock():
 d=json.loads((R/'lock.json').read_text())
 for group in ('gen2','latest'):
  for c in d[group].values():assert sha(c['path'])==c['sha256']
 assert sha(e.BASE)==d['teacher']['sha256'];return d

def bank(phase):
 q.R=R
 if (R/f'{phase}-prefixes.json').exists():return read_prefixes(R/f'{phase}-prefixes.json')
 return q.fixture(phase,1650000 if phase=='validation' else 1660000,32 if phase=='validation' else 96)

def match(path,left,right,f):
 if path.exists():
  r=json.loads(path.read_text());assert r['prefix_sha256']==f['sha256'] and r['left_sha256']==sha(left);assert r['right']==str(right)
  assert r['right_sha256']==sha(right if Path(right).is_file() else e.BASE);return r
 with cold_factory():r=e.match(str(left),str(right),f)
 if right=='teacher100':r['right_sha256']=sha(e.BASE)
 r['cold_per_game']=True;write(path,r);print(path.name,r['summary'],flush=True);return r

def audit_train():
 q.R=R;q.audit_train(base=BASE)
 exclusion_audit()

def exclusion_audit():
 q.R=R
 verified=json.loads((R/'training-audit.json').read_text());assert verified['pass_'];checks={r['seed']:r for r in verified['seeds']}
 # Count only NEW trajectories for exclusion, not pre-existing historical training.
 from .ppo_scale import history
 import torch
 frozen=read_prefixes(R/'forbidden-prefixes.json');starts={state_hash(prefix_state(p)) for p in frozen['prefixes']};reports=[]
 for seed in SEEDS:
  p=q.paths(seed)[-1];assert sha(p)==checks[seed]['checkpoint_sha256'];c=torch.load(p,map_location='cpu',weights_only=False);gs=history(c,p)[4096*16:];visited=set();prefix_hits=0;rejections=0
  for g in gs:
   s=State();rejections+=g.get('prefix_rejections',0)
   for row in g['prefix']:
    play(s,Move(*row) if row else None,validate=False);prefix_hits+=state_hash(s) in starts
   for row in g['moves']:
    visited.add(state_hash(s));play(s,Move(*row) if row else None,validate=False)
   visited.add(state_hash(s))
  assert prefix_hits==0
  reports.append(dict(seed=seed,new_games=len(gs),prefix_rejections=rejections,prefix_start_hits=prefix_hits,incidental_actual_visited_start_overlap=len(visited&starts)))
 write(R/'training-exclusion-audit.json',dict(pass_=True,rows=reports,note='Historical training may precede old evaluations. Exclusion is audited for new games only. Incidental rollout visits are recorded separately from prefix rejection.'))


def validation():
 d=lock();f=bank('validation');selected={}
 for seed in SEEDS:
  key=str(seed);options=[('gen2',Path(d['gen2'][key]['path']))]
  if d['latest'][key]['path']!=str(options[0][1]):options.append(('latest4096',Path(d['latest'][key]['path'])))
  q.R=R;options.extend((p.stem,p) for p in q.paths(seed));candidates=[]
  for label,path in options:
   rows=[]
   for op,spec in [('gen2',d['gen2'][key]['path']),('oldvalue','oldvalue'),('teacher100','teacher100')]:rows.append(match(R/f'seed-{seed}/validation-{label}-{op}.json',path,spec,f))
   if seed==1401:match(R/f'seed-{seed}/validation-{label}-latest4096.json',path,d['latest'][key]['path'],f)
   step=int(path.stem.split('-')[-1]);criterion=[sum(w*r['summary'][k] for w,r in zip((.5,.25,.25),rows)) for k in ('point_rate','mean_difference')]+[-step]
   candidates.append(dict(path=str(path),sha256=sha(path),iteration=step,label=label,criterion=criterion))
  write(R/f'seed-{seed}/candidates.json',candidates);selected[key]=max(candidates,key=lambda x:x['criterion'])
 write(R/'selection.json',selected)

def holdout():
 d=lock();chosen=json.loads((R/'selection.json').read_text());f=bank('holdout')
 assert sha(R/'selection.json')==json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']
 for seed,c in chosen.items():
  assert sha(c['path'])==c['sha256']
  for op,spec in [('gen2',d['gen2'][seed]['path']),('teacher100','teacher100')]+([('latest4096',d['latest'][seed]['path'])] if seed=='1401' else []):match(R/f'seed-{seed}/holdout-{op}.json',c['path'],spec,f)

def audit_eval():
 d=lock();q.R=R;training=q.hashes('training-states');selection=json.loads((R/'selection.json').read_text());phase_reports={};individual={}
 for phase in ('validation','holdout'):
  f=read_prefixes(R/f'{phase}-prefixes.json');lookup={p['id']:p for p in f['prefixes']};pre=json.loads((R/f'{phase}-preflight.json').read_text());assert pre['prefix_sha256']==f['sha256'] and pre['training_states_sha256']==sha(R/'training-states/index.json')
  starts={state_hash(prefix_state(p)) for p in f['prefixes']};assert not starts&training
  for src in pre['sources']:
   assert sha(src['path'])==src['sha256'];old=json.loads(Path(src['path']).read_text());assert not starts&{state_hash(prefix_state(p)) for p in old['prefixes']};assert not set(lookup)&{p['id'] for p in old['prefixes']}
  count=0;visited=set()
  for path in sorted(R.glob(f'seed-*/{phase}-*.json')):
   r=json.loads(path.read_text());assert r['prefix_sha256']==f['sha256'];assert r['summary']==summarize(r['games']);assert sha(r['left'])==r['left_sha256'];assert r['right_sha256']==sha(r['right'] if Path(r['right']).is_file() else e.BASE)
   if phase=='holdout':assert r['left_sha256']==selection[path.parent.name.split('-')[-1]]['sha256']
   seats={};groups={};calls=0
   for g in r['games']:
    s=prefix_state(lookup[g['prefix_id']]);seats.setdefault(g['prefix_id'],[]).append(g['seat']);groups.setdefault(g['prefix_id'],[]).append(g['difference'])
    for row in g['moves']:
     visited.add(state_hash(s));m=Move(*row) if row else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);calls+=1
    visited.add(state_hash(s));assert s.done and g['scores']==[s.score(0),s.score(1)] and g['difference']==s.score(g['seat'])-s.score(1-g['seat']);count+=1
   assert len(seats)==f['count'] and all(sorted(v)==[0,1] for v in seats.values());assert calls==sum(t['calls'] for t in r['timing']);assert r['cold_per_game']
   if phase=='holdout':
    x=np.array([[np.mean([float(z>0)+.5*(z==0) for z in groups[p['id']]]),np.mean(groups[p['id']])] for p in f['prefixes']]);individual[str(path.relative_to(R))]=dict(summary=r['summary'],clustered=clustered(x),timing=[{k:v for k,v in t.items() if k!='all_ms'} for t in r['timing']])
  assert not visited&training,'Do not replace evaluation prefixes after overlap'
  phase_reports[phase]=dict(games=count,unique_actual_states=len(visited),training_actual_overlap=len(visited&training),adopted_start_overlap=0)
 assert sha(R/'selection.json')==json.loads((R/'holdout-preflight.json').read_text())['selection_sha256']
 for seed,c in selection.items():
  options=json.loads((R/f'seed-{seed}/candidates.json').read_text());assert c==max(options,key=lambda x:x['criterion'])
  for option in options:
   rows=[json.loads((R/f"seed-{seed}/validation-{option['label']}-{op}.json").read_text()) for op in ('gen2','oldvalue','teacher100')];assert option['criterion']==[sum(w*r['summary'][k] for w,r in zip((.5,.25,.25),rows)) for k in ('point_rate','mean_difference')]+[-option['iteration']]
 aggregate={}
 for op in ('gen2','teacher100'):
  arrays=[];times=[[],[]]
  for seed in SEEDS:
   r=json.loads((R/f'seed-{seed}/holdout-{op}.json').read_text());groups={}
   for g in r['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
   arrays.append(np.array([[np.mean([float(z>0)+.5*(z==0) for z in groups[p['id']]]),np.mean(groups[p['id']])] for p in read_prefixes(R/'holdout-prefixes.json')['prefixes']]))
   for i in (0,1):times[i]+=r['timing'][i]['all_ms']
  v=clustered(np.mean(arrays,axis=0));v['per_seed_mean']=np.mean(arrays,axis=1).tolist();v['timing']=[dict(mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=max(ts),budget_ms=100 if i==1 and op=='teacher100' else None,over_budget_rate=float(np.mean(np.array(ts)>100)) if i==1 and op=='teacher100' else None) for i,ts in enumerate(times)];aggregate[op]=v
 write(R/'evaluation-audit.json',dict(pass_=True,phases=phase_reports,individual=individual,aggregate=aggregate,note='Fixed selected models, seats and seeds clustered within prefix. Seed1401 latest4096 supplement was preregistered; no holdout reselection.'))
 print(aggregate,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','audit_train','exclusion_audit','validation','holdout','audit_eval','match']);p.add_argument('--left');p.add_argument('--right');p.add_argument('--fixture');p.add_argument('--output');a=p.parse_args()
 if a.mode=='match':
  if not all((a.left,a.right,a.fixture,a.output)):p.error('match requires left/right/fixture/output')
  match(Path(a.output),a.left,a.right,read_prefixes(a.fixture))
 else:globals()[a.mode]()
