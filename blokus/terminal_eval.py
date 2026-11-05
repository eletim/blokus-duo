"""Fixed validation opponents, prefix-paired uncertainty, network-only evaluation."""
import argparse,json,random,time,gzip
from pathlib import Path
import numpy as np
from .terminal_ppo import TerminalAgent,ROOT
from .engine import legal_moves,play
from .ai import features,heuristic,seed_all
from .tournament import make_prefixes,prefix_state,summarize,read_prefixes,digest
from .visited_data import state_hash
from .value_experiment import write
from .value_search import BudgetAgent
from .aligned_data import BASE
from .reply_search import reset_game
from .research import sha

class Heuristic:
 def __init__(self):self.times=[]
 def act(self,s,rng):
  t=time.perf_counter();qs=legal_moves(s)
  if qs:
   scores=heuristic(features(s,qs)[0]);m=qs[int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)))].move
  else:m=None
  self.times.append(time.perf_counter()-t);return m

def agent(spec):
 if spec=='heuristic':return Heuristic()
 if spec=='oldvalue':return BudgetAgent('value',BASE,.5,24,6,'cuda')
 return TerminalAgent(spec)

def match(left,right,fixture):
 a=agent(left);b=agent(right);games=[];start=time.perf_counter();times=[[],[]]
 for i,p in enumerate(fixture['prefixes']):
  for seat in [0,1]:
   for ag in [a,b]:
    if isinstance(ag,BudgetAgent):reset_game(ag)
   s=prefix_state(p);rng=[random.Random(1450000+i*2+j) for j in [0,1]];history=[]
   while not s.done:
    k=0 if s.turn==seat else 1;ag=[a,b][k];t=time.perf_counter();m=ag.act(s,rng[s.turn]);times[k].append((time.perf_counter()-t)*1000);history.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   games.append(dict(prefix_id=p['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=history))
 return dict(left=str(left),right=str(right),left_sha256=sha(left) if Path(left).is_file() else None,right_sha256=sha(right) if Path(right).is_file() else sha(BASE) if right=='oldvalue' else None,prefix_sha256=fixture['sha256'],summary=summarize(games),games=games,seconds=time.perf_counter()-start,timing=[dict(calls=len(ts),mean_ms=float(np.mean(ts)),p95_ms=float(np.quantile(ts,.95)),max_ms=max(ts),all_ms=ts) for ts in times],network_only_left=True,equal_thinking_time=False)

def validate(root=ROOT,prefix_seed=1430000,steps=(16,32,64)):
 root=Path(root);write(root/'eval-config.json',dict(steps=list(steps),prefix_seed=prefix_seed));seed_all(1400);fixture=make_prefixes(prefix_seed,32);write(root/'validation-prefixes.json',fixture);selected={};allrows=[]
 for seed in [1401,1402,1403]:
  out=root/f'seed-{seed}';candidates=[]
  for step in steps:
   path=out/f'iteration-{step}.pt'
   if not path.exists():continue
   rows=[]
   for opponent,spec in [('heuristic','heuristic'),('untrained',str(out/'untrained.pt'))]:
    dest=out/f'validation-{step}-{opponent}.json'
    r=json.loads(dest.read_text()) if dest.exists() else match(str(path),spec,fixture);write(dest,r);rows.append(r);allrows.append(r);print(seed,step,opponent,r['summary'],flush=True)
   criterion=[float(np.mean([r['summary']['point_rate'] for r in rows])),float(np.mean([r['summary']['mean_difference'] for r in rows])),-step];candidates.append(dict(step=step,path=str(path),sha256=sha(path),criterion=criterion,rows=rows))
  assert candidates;chosen=max(candidates,key=lambda r:r['criterion']);selected[str(seed)]=chosen
 # Average models inside each shared prefix, never count paired games/seeds as independent prefixes.
 checks={}
 for oi,opponent in enumerate(['heuristic','untrained']):
  arrays=[]
  for seed,c in selected.items():
   groups={}
   for g in c['rows'][oi]['games']:groups.setdefault(g['prefix_id'],[]).append(g['difference'])
   arrays.append(np.array([[np.mean([float(d>0)+.5*float(d==0) for d in ds]),np.mean(ds)] for ds in groups.values()]))
  x=np.mean(arrays,axis=0);ix=np.random.default_rng(1460000).integers(len(x),size=(4000,len(x)));ci=np.quantile(x[ix].mean(1),[.025,.975],axis=0);means=np.mean(arrays,axis=1)
  checks[opponent]=dict(point_rate=float(x[:,0].mean()),difference=float(x[:,1].mean()),point_ci95=ci[:,0].tolist(),difference_ci95=ci[:,1].tolist(),each_seed_nonnegative=bool((means[:,0]>=.5).all()),advance=bool((means[:,0]>=.5).all() and x[:,0].mean()>=.53 and x[:,1].mean()>0 and ci[0,0]>.5))
 metadata={s:{k:v for k,v in c.items() if k!='rows'} for s,c in selected.items()};write(root/'selection.json',metadata);decision=dict(advance=all(c['advance'] for c in checks.values()),checks=checks,criterion='preregistered win/loss validation gate; no teacher MSE',prefix_sha256=fixture['sha256']);write(root/'decision.json',decision);print(decision,flush=True)
 # Strong reference is explicitly excluded from selection and gate.
 for seed,c in selected.items():
  dest=root/f'seed-{seed}/validation-selected-oldvalue.json';r=match(c['path'],'oldvalue',fixture);write(dest,r);print(seed,'oldvalue',r['summary'],flush=True)
 return decision

def disjoint_fixture(seed,count,forbidden_prefixes,forbidden_states):
 selected=[];seen=set()
 for p in make_prefixes(seed,count*4)['prefixes']:
  h=state_hash(prefix_state(p))
  if p['id'] in forbidden_prefixes or h in forbidden_states or h in seen:continue
  selected.append(p);seen.add(h)
  if len(selected)==count:break
 if len(selected)!=count:raise ValueError('finite prefix candidate pool exhausted')
 return dict(format=1,seed=seed,count=count,sha256=digest(selected),prefixes=selected)

def training_seen(root):
 prefixes=set();states=set()
 for directory in Path(root).glob('seed-*'):
  path=directory/'games.json';compressed=directory/'games.json.gz'
  if not path.exists() and not compressed.exists():raise ValueError('missing replay data for holdout separation')
  games=json.loads(path.read_text()) if path.exists() else json.load(gzip.open(compressed,'rt'))
  for g in games:
   prefixes.add(digest(g['prefix']));states.update(r['state_hash'] for r in g['targets'])
 return prefixes,states

def holdout(root=ROOT,prefix_seed=1440000,fixture_path=None):
 root=Path(root);assert json.loads((root/'decision.json').read_text())['advance'];selection=json.loads((root/'selection.json').read_text())
 forbidden,states=training_seen(root);forbidden.update(p['id'] for p in read_prefixes(root/'validation-prefixes.json')['prefixes'])
 fixture=read_prefixes(fixture_path) if fixture_path else disjoint_fixture(prefix_seed,96,forbidden,states)
 starts={state_hash(prefix_state(p)) for p in fixture['prefixes']}
 if len(starts)!=96 or starts&states or forbidden&{p['id'] for p in fixture['prefixes']}:raise ValueError('holdout collides with training/validation; do not evaluate')
 write(root/'holdout-prefixes.json',fixture)
 for seed,c in selection.items():
  assert sha(c['path'])==c['sha256']
  for name,spec in [('untrained',str(root/f'seed-{seed}/untrained.pt')),('heuristic','heuristic'),('oldvalue','oldvalue')]:
   r=match(c['path'],spec,fixture);write(root/f'seed-{seed}/holdout-{name}.json',r);print(seed,name,r['summary'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','holdout']);p.add_argument('--root',default=str(ROOT));p.add_argument('--prefix-seed',type=int);p.add_argument('--fixture');p.add_argument('--steps',type=int,nargs='+',default=[16,32,64]);a=p.parse_args();fn=validate if a.mode=='validation' else holdout;kw={'root':a.root}
 if a.mode=='validation':kw['steps']=a.steps
 if a.fixture:kw['fixture_path']=a.fixture
 if a.prefix_seed:kw['prefix_seed']=a.prefix_seed
 fn(**kw)
