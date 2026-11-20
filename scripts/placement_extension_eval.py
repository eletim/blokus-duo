"""Preregistered stage trends and unconditional final four-comparison confirmation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,gzip,json,random,time,copy
from pathlib import Path
import numpy as np
import torch
import placement_large_eval as q
from blokus.placement_value import PlacementValue,PlacementEvaluator
from blokus.ppo_search_deeper import DeepSearch
R=Path('artifacts/value-placement-49152');PREV=Path('artifacts/value-placement-24576');e=q.e

def banks():
 e.setup();assert not (R/'evaluation-banks.json').exists();ex=set(q.readgz(R/'exclusions.json.gz')['states'])
 for phase in ('train','dev'):ex.update(q.readgz(R/f'{phase}-visited.json.gz')['states'])
 with gzip.open(R/'evaluation-excluded.json.gz','wt') as h:json.dump(dict(states=sorted(ex)),h)
 hashes={}
 for count,phase,n,base in [(36864,'validation',8,886400000000),(49152,'validation',8,886500000000),(49152,'confirmation',64,886600000000)]:
  rng=random.Random(base);bank=[]
  for i in range(n):pr=e.prefix(rng,np.load(e.INDEX,mmap_mode='r'),ex,8+i%3*2);ex.add(pr['state_hash']);bank.append(pr)
  dest=R/f'n{count}'/f'{phase}-bank.json';e.write(dest,dict(prefixes=bank));hashes[str(dest)]=e.sha(dest)
 e.write(R/'evaluation-banks.json',hashes)

def source(name,stage,seed):
 if name=='F':return Path(f'artifacts/ppo-gen3/seed-{seed}.pt')
 if name=='Q':return Path(f'artifacts/value-placement-scale/n3072/A-{seed}.pt')
 if name=='R':return PREV/f'A-{seed}.pt'
 if name=='P':return (PREV if stage==36864 else R/'n36864')/f'A-{seed}.pt'
 return R/f'n{stage}'/f'{name}-{seed}.pt'
def agent(name,stage,p):
 if name=='F':return DeepSearch(p,500,max_depth=6)
 ck=torch.load(source(name,stage,p.model_seed),map_location='cpu',weights_only=True);assert ck['source_sha']==e.sha(f'artifacts/ppo-gen3/seed-{p.model_seed}.pt')
 m=PlacementValue(p.model,ck['variant']);m.load_state_dict(ck['model']);return DeepSearch(p,500,max_depth=6,value_evaluator=PlacementEvaluator(m))
def aggregate(stage,phase,pair,family):
 q.R=R/f'n{stage}';out=q.aggregate(phase,pair)
 rows=[g for seed in e.SEEDS for g in q.readgz(q.R/f'{phase}-{pair}-{seed}.json.gz')['games']];groups=sorted({g['prefix'] for g in rows});ix=np.random.default_rng(771).integers(len(groups),size=(20000,len(groups)))
 for key in ('point','difference'):
  x=np.array([np.mean([g[key] for g in rows if g['prefix']==i]) for i in groups]);boot=x[ix].mean(1);out[key+'_adjusted_ci']=np.quantile(boot,[.05/(2*family),1-.05/(2*family)]).tolist()
 return out

def evaluate(stage,phase):
 os.sched_setaffinity(0,{0});root=R/f'n{stage}';q.R=e.R=root;e.setup();start=time.perf_counter();warmups=[];pairs=['AP','AF'] if phase=='validation' else ['AQ','AR','AF','AC'];assert phase=='validation' or stage==49152
 for p,h in json.loads((R/'evaluation-banks.json').read_text()).items():assert e.sha(p)==h
 bank=json.loads((root/f'{phase}-bank.json').read_text())['prefixes'];assert len(bank)==(8 if phase=='validation' else 64)
 manifest={str(source(name,stage,s)):e.sha(source(name,stage,s)) for name in set(''.join(pairs)) for s in e.SEEDS};manifest[str(root/'checkpoint-selection.json')]=e.sha(root/'checkpoint-selection.json');manifest[str(root/f'{phase}-bank.json')]=e.sha(root/f'{phase}-bank.json')
 mp=root/f'{phase}-models.json'
 if mp.exists():assert json.loads(mp.read_text())==manifest
 else:e.write(mp,manifest)
 for pair in pairs:
  left,right=pair
  for seed in e.SEEDS:
   dest=root/f'{phase}-{pair}-{seed}.json.gz'
   if dest.exists():
    d=q.readgz(dest);assert d['seed']==seed and d['left']==left and d['right']==right and len(d['games'])==2*len(bank);continue
   p=e.PPOAgent(seed,'cpu',backend='native');aa=agent(left,stage,p);bb=agent(right,stage,p);games=[]
   for a in (aa,bb):
    t=time.perf_counter();a.act(e.state(bank[0]['moves']),random.Random(2));warmups.append(dict(pair=pair,seed=seed,seconds=time.perf_counter()-t))
   for pi,pr in enumerate(bank):
    for seat in (0,1):
     z=e.state(pr['moves']);moves=[];stats=[];rng=random.Random(seed*100+pi*2+seat)
     while not z.done:
      a=aa if z.turn==seat else bb;m=a.act(z,rng);stats.append(dict(player=z.turn,method=left if z.turn==seat else right,**copy.deepcopy(a.stats)));q.play(z,m,validate=True);moves.append(e.raw(m))
     scores=[z.score(0),z.score(1)];diff=scores[seat]-scores[1-seat];games.append(dict(prefix=pi,seat=seat,moves=moves,scores=scores,difference=diff,point=float(diff>0)+.5*(diff==0),stats=stats));print('EVAL',stage,phase,pair,seed,pi,seat,round(time.perf_counter()-start),flush=True)
   tmp=dest.with_suffix('.tmp')
   with gzip.open(tmp,'wt') as h:json.dump(dict(left=left,right=right,seed=seed,budget_ms=500,max_depth=6,games=games),h)
   tmp.replace(dest)
 out={pair:aggregate(stage,phase,pair,len(pairs)) for pair in pairs};e.write(root/f'{phase}-summary.json',out)
 if phase=='confirmation':
  passed=all(v['point_adjusted_ci'][0]>.5+1e-9 and v['difference']>0 and v['stats']['A']['over_550']<=.01 for v in out.values());e.write(root/'decision.json',dict(passed=passed,results=out))
 e.write(root/f'{phase}-time.json',dict(seconds=time.perf_counter()-start,warmups=warmups,resumed_note='outer invocation wall; completed shards retained'))
 print('EVALUATED',stage,phase,{p:v['point'] for p,v in out.items()},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['banks','validation','confirmation']);p.add_argument('--stage',type=int,choices=[36864,49152],default=49152);a=p.parse_args();banks() if a.command=='banks' else evaluate(a.stage,a.command)
