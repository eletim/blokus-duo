"""Fixed L confirmation; no training, selection, or changes to the tuple gate."""
import argparse,gzip,hashlib,json,random,time
from pathlib import Path
import numpy as np
from blokus.engine import Move,play
from blokus.linear_rl import prefix
from blokus.linear_train2 import state_hash
from blokus.linear_value import Agent,Search
from blokus.linear_tree import CompletedSearch
from scripts import linear_value_eval as base,linear_search_run as search,tuple_cycle as cycle
R=Path('artifacts/linear-L-confirm');SEEDS=(3101,3102,3103)
def write(name,d):(R/name).write_text(json.dumps(d,indent=2)+'\n')
def read(p):
 if str(p).endswith('.gz'):
  with gzip.open(p,'rt') as f:return json.load(f)
 return json.loads(Path(p).read_text())
def source(label,seed):return Path(f'artifacts/linear-tuple/L-{seed}.json') if label=='L' else Path(f'artifacts/linear-hybrid/exports/seed-{seed}.json') if label=='F' else Path(f'artifacts/ppo-gen3/seed-{1751401+seed-3101}.pt')
def indices():return cycle.indices()+[np.load('artifacts/linear-tuple/new-visited.npy',mmap_mode='r')]
def make(label,seed):return base.make('gen3',seed) if label=='gen3' else CompletedSearch(Agent.load(source(label,seed)),ms=100,width=24,replies=8,depth=2)
def check_models():
 for d in read(R/'models.json'):assert base.sha(d['path'])==d['sha256']
def setup():
 import torch
 torch.set_num_threads(1);check_models()
def prepare():
 models=[]
 for label in ('L','F','gen3'):
  for seed in SEEDS:
   p=source(label,seed);d=dict(label=label,seed=seed,path=str(p),sha256=base.sha(p))
   if label!='gen3':
    a=Agent.load(p);d.update(policy_sha256=hashlib.sha256(a.weights.tobytes()).hexdigest(),value_sha256=hashlib.sha256(a.value_weights.tobytes()).hexdigest())
   models.append(d)
 assert all(models[i]['policy_sha256']==models[i+3]['policy_sha256'] for i in range(3))
 write('models.json',models)
 forbidden=set(read('artifacts/linear-tuple/exclusions.json')['states']);oldvisits=set();sources=[]
 for folder in Path('artifacts').glob('linear-*'):
  if folder==R:continue
  for p in folder.glob('*-prefixes.json'):
   forbidden.update(x['hash'] for x in read(p).get('prefixes',[]))
  for pattern in ('validation-*.json*','confirmation-*.json*'):
   for p in folder.glob(pattern):
    if p.suffix not in ('.json','.gz'):continue
    d=read(p)
    if not isinstance(d,dict) or not isinstance(d.get('games'),list):continue
    sources.append(dict(path=str(p),sha256=base.sha(p),games=len(d['games'])))
    for g in d['games']:
     s=prefix(g['prefix_seed']);oldvisits.add(state_hash(s))
     for row in g['moves']:
      play(s,Move(*row) if row else None,validate=False);oldvisits.add(state_hash(s))
 old=np.array(sorted(bytes.fromhex(h) for h in oldvisits),dtype='S32');np.save(R/'old-evaluation-states.npy',old)
 write('old-evaluation.json',dict(sources=sources,states=len(old),raw_sha256=hashlib.sha256(old.tobytes()).hexdigest(),excluded_starts=sorted(forbidden)))
 idx=indices()+[old];rows=[];i=0
 while len(rows)<256:
  ps=2480000000+i;i+=1;h=state_hash(prefix(ps))
  if h in forbidden or any(base.contains(a,h) for a in idx):continue
  forbidden.add(h);rows.append(dict(seed=ps,hash=h))
 write('prefixes.json',dict(count=256,rejected=i-256,prefixes=rows,sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()))
 write('pre-audit.json',dict(pass_=True,models=len(models),prefixes=256,old_evaluation_states=len(old),indices=[dict(states=len(a),raw_sha256=hashlib.sha256(memoryview(a).cast('B')).hexdigest()) for a in idx],plan_sha256=base.sha(R/'PLAN.md')))
 print('PREPARED',flush=True)
def interval(games,group_size):
 groups={}
 for g in games:groups.setdefault(g['prefix_seed'],[]).append(g)
 assert all(len(v)==group_size for v in groups.values())
 p=np.array([np.mean([g['point'] for g in v]) for v in groups.values()]);d=np.array([np.mean([g['difference'] for g in v]) for v in groups.values()]);ix=np.random.default_rng(2490000000).integers(len(p),size=(20000,len(p)));pp=p[ix].mean(1);dd=d[ix].mean(1)
 return dict(point=float(p.mean()),point_ci95=np.quantile(pp,[.025,.975]).tolist(),point_ci_bonferroni2=np.quantile(pp,[.0125,.9875]).tolist(),difference=float(d.mean()),difference_ci95=np.quantile(dd,[.025,.975]).tolist(),wins=sum(g['difference']>0 for g in games),draws=sum(g['difference']==0 for g in games),losses=sum(g['difference']<0 for g in games),games=len(games))
def summarize(ds):
 games=sum([d['games'] for d in ds],[]);r=interval(games,6);r.update(by_seed={str(d['seed']):interval(d['games'],2) for d in ds},by_seat={str(seat):interval([g for g in games if g['seat']==seat],3) for seat in (0,1)},left_ms=float(np.mean([d['left_ms'] for d in ds])),right_ms=float(np.mean([d['right_ms'] for d in ds])),wall_seconds=sum(d['wall'] for d in ds));return r

def match(right,seed,bank):
 path=R/f'L-{right}-{seed}.json.gz'
 if path.exists():
  d=read(path);assert d['prefix_sha256']==bank['sha256'] and d['left_model_sha256']==base.sha(source('L',seed)) and d['right_model_sha256']==base.sha(source(right,seed));return d
 bots=[];cold=[]
 for label in ('L',right):
  t=time.perf_counter();a=make(label,seed);made=time.perf_counter();a.act(prefix(bank['prefixes'][0]['seed']),random.Random(0));cold.append(dict(label=label,load_ms=(made-t)*1000,first_act_ms=(time.perf_counter()-made)*1000));bots.append(a)
  if isinstance(a,Search):a.stats.clear()
 start=time.perf_counter();games=[];times=[[],[]]
 for row in bank['prefixes']:
  for seat in (0,1):
   s=prefix(row['seed']);rng=random.Random(row['seed']*2+seat);moves=[]
   while not s.done:
    side=int(s.turn!=seat);begin=time.perf_counter();m=bots[side].act(s,rng);times[side].append((time.perf_counter()-begin)*1000);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   diff=s.score(seat)-s.score(1-seat);games.append(dict(prefix_seed=row['seed'],seat=seat,moves=moves,scores=[s.score(0),s.score(1)],difference=diff,point=float(np.sign(diff)*.5+.5)))
 d=dict(left='L',right=right,seed=seed,prefix_sha256=bank['sha256'],left_model_sha256=base.sha(source('L',seed)),right_model_sha256=base.sha(source(right,seed)),cold=cold,warmup_excluded=True,per_act_cost_prediction_reset=True,games=games,wall=time.perf_counter()-start,left_ms=float(np.mean(times[0])),right_ms=float(np.mean(times[1])),left_search=bots[0].stats,right_search=bots[1].stats if isinstance(bots[1],Search) else [])
 path.write_bytes(gzip.compress(json.dumps(d).encode(),mtime=0));print('FINISHED',right,seed,len(games),round(d['wall'],1),flush=True);return d

def evaluate():
 setup();bank=read(R/'prefixes.json');assert bank['count']==256;summary={};costs={};start=time.perf_counter()
 for right in ('F','gen3'):
  ds=[match(right,s,bank) for s in SEEDS];summary['L/'+right]=summarize(ds);costs['L/'+right]={side:search.summarize_stats(sum([d[side+'_search'] for d in ds],[])) for side in ('left','right') if ds[0][side+'_search']}
  write('results.json',summary);write('costs.json',costs)
 check_models();write('decision.json',dict(complete=True,old_nonNN_superiority=summary['L/F']['point_ci_bonferroni2'][0]>.5,gen3_superiority=summary['L/gen3']['point_ci_bonferroni2'][0]>.5,wall_seconds=time.perf_counter()-start,models_changed=False,training=False,old_gate_unchanged=True));print('COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);a=p.parse_args();prepare() if a.mode=='prepare' else evaluate()
