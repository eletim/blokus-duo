"""Finite frozen PPO search experiment, isolated timing phases and exact replay audits."""
import argparse,json,time,random,os
from pathlib import Path
import numpy as np
from . import ppo_curve_eval as q,terminal_eval as e
from .ppo_generation_eval import clustered
from .ppo_short_search import ShortSearch
from .terminal_ppo import observe,collate
from .engine import Move,play,is_legal,legal_moves
from .research import sha,clone
from .visited_data import state_hash
from .tournament import prefix_state,read_prefixes,summarize
from .value_experiment import write
from .refinement_search import RefinementAgent
R=Path('artifacts/ppo-short-search');SEEDS=(1401,1402,1403)

def prepare():
 if (R/'lock.json').exists():return
 models=json.loads(Path('artifacts/ppo-gen2/manifest.json').read_text())
 for c in models.values():assert sha(c['path'])==c['sha256']
 for name in ('training-states','training-prefixes'):
  folder=R/name;base=Path('artifacts/ppo-generation-pool')/name/'index.json';index=json.loads(base.read_text());write(folder/'index.json',dict(count=index['count'],parts=[],bases=[dict(path=os.path.relpath(base,folder),sha256=sha(base))]))
 write(R/'lock.json',dict(models=models,teacher_sha256=sha(e.BASE),plan_sha256=sha(R/'PLAN.md'),started=time.time(),training_updates=0))

def bank(phase):
 q.R=R;return q.fixture(phase,{'dev':1690000,'validation':1700000,'holdout':1710000}[phase],{'dev':36,'validation':32,'holdout':96}[phase])

def model(seed):
 c=json.loads((R/'lock.json').read_text())['models'][str(seed)];assert sha(c['path'])==c['sha256'];return c['path']

def cold(ag):
 if isinstance(ag,e.BudgetAgent):e.reset_game(ag)
 if hasattr(ag,'reset_game'):ag.reset_game()

def stats(records):
 nonforced=[r for r in records if not r['forced']]
 return dict(calls=len(records),nonforced=len(nonforced),completed=sum(r['completed'] for r in records),fallback_rate=sum(r['fallback'] for r in nonforced)/max(1,len(nonforced)),switch_rate=sum(r['switched'] for r in records)/max(1,len(records)),overrun_rate=np.mean([r['overrun'] for r in records]).item(),mean_roots=np.mean([r['roots'] for r in records]).item(),mean_replies=np.mean([r['replies'] for r in records]).item(),phase_mean_ms={k:np.mean([r[k] for r in records]).item() for k in ['root_ms','prepare_ms','policy_ms','leaf_ms']})

def match(seed,budget,margin,opponent,f,dest):
 if dest.exists():return json.loads(dest.read_text())
 a=ShortSearch(model(seed),budget,margin);b=RefinementAgent(e.BASE,100,'priority','cuda') if opponent=='teacher100' else e.agent('oldvalue' if opponent=='oldvalue' else model(seed));games=[];times=[[],[]];records=[];start=time.perf_counter()
 for i,p in enumerate(f['prefixes']):
  for seat in [0,1]:
   cold(a);cold(b);s=prefix_state(p);rng=[random.Random(1720000+i*2+j) for j in [0,1]];history=[]
   while not s.done:
    k=0 if s.turn==seat else 1;ag=[a,b][k];t=time.perf_counter();m=ag.act(s,rng[s.turn]);times[k].append((time.perf_counter()-t)*1000);history.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   records.extend(a.records);games.append(dict(prefix_id=p['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=history))
 r=dict(seed=seed,budget_ms=budget,margin=margin,opponent=opponent,model_sha256=sha(model(seed)),opponent_sha256=sha(e.BASE) if opponent in ('oldvalue','teacher100') else sha(model(seed)),prefix_sha256=f['sha256'],games=games,summary=summarize(games),seconds=time.perf_counter()-start,records=records,search=stats(records),cold_per_game=True,equal_thinking_time=False,timing=[dict(calls=len(ts),mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=max(ts),overrun_rate=float(np.mean(np.array(ts)>limit)) if limit else None,all_ms=ts) for ts,limit in zip(times,[budget,100 if opponent=='teacher100' else 6 if opponent=='oldvalue' else None])]);write(dest,r);print(dest.name,r['summary'],r['search'],flush=True);return r

def aggregate(rows):
 arrays=[]
 for r in rows:
  groups={}
  for g in r['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
  arrays.append(np.array([[np.mean([float(x>0)+.5*float(x==0) for x in ds]),np.mean(ds)] for ds in groups.values()]))
 return clustered(np.mean(arrays,axis=0))

def dev():
 f=bank('dev');rows=[];roots=[];profile={6:[],25:[]};i=0
 for seed in SEEDS:
  a=e.agent(model(seed));search={budget:ShortSearch(model(seed),budget) for budget in [6,25]}
  for opponent in ['self','heuristic','oldvalue']:
   b=e.agent(model(seed) if opponent=='self' else opponent)
   for rep in range(4):
    p=f['prefixes'][i];i+=1;s=prefix_state(p);cold(a);cold(b);rng=random.Random(1691000+i);history=[];local=[];ply=len(p['moves'])
    while not s.done:
     if ply in [8,20,32]:
      o,_=observe(s)
      import torch
      with torch.no_grad():_,v=a.model(*collate([o],a.device))
      local.append(dict(game=i,ply=ply,player=s.turn,value=float(v.item()),state_hash=state_hash(s)))
      for budget,ag in search.items():
       cold(ag);ag.act(s,random.Random(1692000+i+ply));profile[budget].extend(ag.records)
     m=(a if s.turn==0 else b).act(s,rng);history.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False);ply+=1
    for r in local:r['return']=float(np.sign(s.score(r['player'])-s.score(1-r['player'])));r['opponent']=opponent;r['seed']=seed
    roots.extend(local);rows.append(dict(game=i,prefix_id=p['id'],seed=seed,opponent=opponent,scores=[s.score(0),s.score(1)],moves=history))
 calibration={}
 for opponent in ['self','heuristic','oldvalue']:
  selected=[r for r in roots if r['opponent']==opponent];game_mse=[]
  for game in sorted({r['game'] for r in selected}):game_mse.append(np.mean([(r['value']-r['return'])**2 for r in selected if r['game']==game]))
  arr=np.array(game_mse);ix=np.random.default_rng(1693000).integers(len(arr),size=(4000,len(arr)));calibration[opponent]=dict(games=len(arr),roots=len(selected),mse=float(arr.mean()),game_bootstrap_mse_ci95=np.quantile(arr[ix].mean(1),[.025,.975]).tolist(),zero_mse=float(np.mean([r['return']**2 for r in selected])))
 write(R/'dev.json',dict(games=rows,roots=roots,calibration=calibration,profile={str(k):dict(summary=stats(v),records=v,mean_ms=float(np.mean([r['elapsed_ms'] for r in v]))) for k,v in profile.items()},conditional_diagnostic_not_true_value=True));print(calibration,flush=True)

def validation(margin):
 f=bank('validation');options=[]
 for budget in [6,25]:
  rows=[match(seed,budget,margin,'self',f,R/f'validation-{margin}-{budget}-{seed}.json') for seed in SEEDS];agg=aggregate(rows);fallback=sum(sum(x['fallback'] for x in r['records'] if not x['forced']) for r in rows)/sum(r['search']['nonforced'] for r in rows)
  # summarize supplies prefix-clustered per-seed point intervals.
  upper=[r['summary']['point_rate_ci95'][1] if 'point_rate_ci95' in r['summary'] else r['summary']['point_ci95'][1] for r in rows]
  advance=agg['point']>=.52 and agg['difference']>0 and min(upper)>=.5 and fallback<.5
  options.append(dict(budget=budget,margin=margin,aggregate=agg,fallback_rate=fallback,advance=advance,seed_summaries=[r['summary'] for r in rows]))
 chosen=max(options,key=lambda x:(x['aggregate']['point'],x['aggregate']['difference'],-x['budget']));write(R/f'validation-decision-{margin}.json',dict(options=options,chosen=chosen));print('DECISION',chosen,flush=True);return chosen

def audit():
 states=q.read_hash_index(R/'training-states/index.json');checks=[]
 for path in sorted(R.glob('*.json')):
  if not path.name.startswith(('validation-','holdout-')):continue
  r=json.loads(path.read_text())
  if 'games' not in r:continue
  phase=path.name.split('-')[0];f=read_prefixes(R/f'{phase}-prefixes.json');prefixes={p['id']:p for p in f['prefixes']};assert r['prefix_sha256']==f['sha256'];assert r['model_sha256']==sha(model(r['seed']));seen=set();seats=set();plies=0;counts=[0,0]
  for g in r['games']:
   pair=(g['prefix_id'],g['seat']);assert pair not in seats;seats.add(pair);s=prefix_state(prefixes[g['prefix_id']])
   for row in g['moves']:
    seen.add(state_hash(s));counts[0 if s.turn==g['seat'] else 1]+=1;m=Move(*row) if row is not None else None;assert is_legal(s,m) if m else not legal_moves(s);play(s,m,validate=False);plies+=1
   seen.add(state_hash(s));assert s.done and [s.score(0),s.score(1)]==g['scores'];assert g['difference']==s.score(g['seat'])-s.score(1-g['seat'])
  assert len(seats)==len(prefixes)*2 and not seen&states;assert summarize(r['games'])==r['summary'];assert counts==[t['calls'] for t in r['timing']];assert counts[0]==len(r['records']);assert stats(r['records'])==r['search'];checks.append(dict(path=str(path),sha256=sha(path),games=len(seats),plies=plies,visited=len(seen),training_overlap=0))
 write(R/'evaluation-audit.json',dict(pass_=True,checks=checks,training_states=len(states)));print('AUDIT',len(checks),flush=True)

def run():
 prepare()
 if not (R/'dev.json').exists():dev()
 first=validation(0.)
 chosen=first
 if not chosen['advance']:
  write(R/'revision.json',dict(reason='preregistered gate failed',first=first,change='require WDL improvement >0.15 to replace policy greedy; no width or checkpoint changes'));chosen=validation(.15)
 write(R/'selection.json',dict(chosen=chosen,models=json.loads((R/'lock.json').read_text())['models'],fixed_before_holdout=True))
 f=bank('validation')
 for seed in SEEDS:match(seed,chosen['budget'],chosen['margin'],'oldvalue',f,R/f'validation-aux-{seed}.json')
 audit()
 if chosen['advance']:
  f=bank('holdout')
  for seed in SEEDS:
   for opponent in ['self','teacher100']:match(seed,chosen['budget'],chosen['margin'],opponent,f,R/f'holdout-{opponent}-{seed}.json')
  audit()
 write(R/'complete.json',dict(complete=True,advance=chosen['advance'],finished=time.time(),learning_updates=0))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','dev','run','audit']);a=p.parse_args();e.seed_all(1690000);globals()[a.mode]()
