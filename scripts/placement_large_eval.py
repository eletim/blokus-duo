"""New CPU-only timed strength comparisons; dev selection is already frozen."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,gzip,json,random,time,copy
from pathlib import Path
import numpy as np
import torch
import placement_experiment as e
from blokus.engine import State,Move,play
from blokus.ppo_runtime import state_hash
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_search_deeper import DeepSearch
R=Path('artifacts/value-placement-24576');OLD=Path('artifacts/value-placement-scale/n3072');PAIRS=[('A','Q'),('A','F'),('A','C')]
def readgz(p):return json.load(gzip.open(p,'rt'))
def banks():
 e.setup();assert not (R/'evaluation-manifest.json').exists();ex=set(readgz(R/'exclusions.json.gz')['states'])
 for p in (R/'train-visited.json.gz',R/'dev-visited.json.gz'):ex.update(readgz(p)['states'])
 ex.update(json.loads(Path('artifacts/value-placement-6144/train-labels.json').read_text())['visited']);idx=np.load(e.INDEX,mmap_mode='r')
 with gzip.open(R/'evaluation-excluded.json.gz','wt') as h:json.dump(dict(states=sorted(ex)),h)
 for phase,n,base in [('validation',16,885400000000),('confirmation',32,885500000000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):pr=e.prefix(rng,idx,ex,8+i%3*2);ex.add(pr['state_hash']);bank.append(pr)
  e.write(R/f'{phase}-bank.json',dict(prefixes=bank))
 files=[str(R/f'{v}-{s}.pt') for v in 'AC' for s in e.SEEDS]+[str(OLD/f'A-{s}.pt') for s in e.SEEDS]+[f'artifacts/ppo-gen3/seed-{s}.pt' for s in e.SEEDS]+[str(R/f'{p}-bank.json') for p in ('validation','confirmation')]+[str(R/'checkpoint-selection.json'),'scripts/placement_large_eval.py','blokus/placement_value.py','blokus/ppo_search_deeper.py']
 e.write(R/'evaluation-manifest.json',{p:e.sha(p) for p in files});print('EVALUATION BANKS FIXED',flush=True)
def agent(name,p,budget=500,depth=6):
 if name=='F':return DeepSearch(p,budget,max_depth=depth)
 src=OLD/f'A-{p.model_seed}.pt' if name=='Q' else R/f'{name}-{p.model_seed}.pt';ck=torch.load(src,map_location='cpu',weights_only=True);assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{p.model_seed}.pt')
 m=PlacementValue(p.model,ck['variant']);m.load_state_dict(ck['model']);return DeepSearch(p,budget,max_depth=depth,value_evaluator=PlacementEvaluator(m))
def aggregate(phase,pair):
 e.R=R;out=e.aggregate(phase,pair);rows=[dict(seed=s,**g) for s in e.SEEDS for g in readgz(R/f'{phase}-{pair}-{s}.json.gz')['games']];groups=sorted({g['prefix'] for g in rows});pt=np.array([np.mean([g['point'] for g in rows if g['prefix']==i]) for i in groups]);df=np.array([np.mean([g['difference'] for g in rows if g['prefix']==i]) for i in groups]);ix=np.random.default_rng(771).integers(len(groups),size=(20000,len(groups)))
 for key,x in [('point',pt),('difference',df)]:
  boot=x[ix].mean(1);out[key+'_ci95']=np.quantile(boot,[.025,.975]).tolist();out[key+'_adjusted_ci']=np.quantile(boot,[.05/6,1-.05/6]).tolist()
 out['by_seat']={str(seat):dict(point=float(np.mean([g['point'] for g in rows if g['seat']==seat])),difference=float(np.mean([g['difference'] for g in rows if g['seat']==seat]))) for seat in (0,1)}
 for name in pair:
  times=[st['elapsed_ms'] for g in rows for st in g['stats'] if st['method']==name and not st['forced_pass']];out['stats'][name].update(p95_ms=float(np.quantile(times,.95)),p99_ms=float(np.quantile(times,.99)))
 return out
def evaluate(phase):
 os.sched_setaffinity(0,{0});e.R=R;e.setup();start=time.perf_counter();warmups=[]
 if phase=='confirmation':assert json.loads((R/'selection.json').read_text())['passed']
 for p,h in json.loads((R/'evaluation-manifest.json').read_text()).items():assert e.sha(p)==h,p
 bank=json.loads((R/f'{phase}-bank.json').read_text())['prefixes']
 for left,right in PAIRS:
  for seed in e.SEEDS:
   dest=R/f'{phase}-{left}{right}-{seed}.json.gz'
   if dest.exists():
    d=readgz(dest);assert d['seed']==seed and d['left']==left and d['right']==right and len(d['games'])==2*len(bank);continue
   p=e.PPOAgent(seed,'cpu',backend='native');aa=agent(left,p);bb=agent(right,p);games=[]
   for a in (aa,bb):
    t=time.perf_counter();a.act(e.state(bank[0]['moves']),random.Random(2));warmups.append(dict(pair=left+right,seed=seed,seconds=time.perf_counter()-t))
   for pi,pr in enumerate(bank):
    for seat in (0,1):
     z=e.state(pr['moves']);moves=[];stats=[];rng=random.Random(seed*100+pi*2+seat)
     while not z.done:
      a=aa if z.turn==seat else bb;m=a.act(z,rng);stats.append(dict(player=z.turn,method=left if z.turn==seat else right,**copy.deepcopy(a.stats)));play(z,m,validate=True);moves.append(e.raw(m))
     scores=[z.score(0),z.score(1)];diff=scores[seat]-scores[1-seat];games.append(dict(prefix=pi,seat=seat,moves=moves,scores=scores,difference=diff,point=float(diff>0)+.5*(diff==0),stats=stats));print('EVAL',phase,left+right,seed,pi,seat,round(time.perf_counter()-start),flush=True)
   tmp=dest.with_suffix('.tmp')
   with gzip.open(tmp,'wt') as h:json.dump(dict(left=left,right=right,seed=seed,budget_ms=500,max_depth=6,games=games),h)
   tmp.replace(dest)
 out={a+b:aggregate(phase,a+b) for a,b in PAIRS};e.write(R/f'{phase}-summary.json',out)
 if phase=='validation':passed=all(2*out[p]['wins']+out[p]['draws']>out[p]['games'] and out[p]['difference']>0 for p in ('AQ','AF')) and 2*out['AC']['wins']+out['AC']['draws']>=out['AC']['games'] and all(out[p]['stats']['A']['over_550']<=.01 for p in out)
 else:passed=all(v['point_adjusted_ci'][0]>.5+1e-9 and v['difference']>0 and v['stats']['A']['over_550']<=.01 for v in out.values())
 e.write(R/('selection.json' if phase=='validation' else 'decision.json'),dict(passed=bool(passed),results=out));e.write(R/f'{phase}-time.json',dict(seconds=time.perf_counter()-start,warmups=warmups,resumed_note='completed pair/seed shards retained; this invocation wall only'))
 print('DECISION',phase,passed,{p:v['point'] for p,v in out.items()},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['banks','validation','confirmation']);a=p.parse_args();banks() if a.command=='banks' else evaluate(a.command)
