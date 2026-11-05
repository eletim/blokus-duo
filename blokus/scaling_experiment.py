"""Finite three-fold/three-seed grid, registered extension and game diagnostic gate."""
import argparse,json,time
import numpy as np
from .scaling_data import ROOT
from .scaling_train import train
from .value_experiment import write
SEEDS=[1101,1102,1103]

def aggregate(size,condition,step=1000):
 rows=[];by={};folds={};seed_results={}
 for f in range(3):
  group=[]
  for seed in SEEDS:
   r=json.loads((ROOT/f'n{size}-f{f}-s{seed}-{condition}/result.json').read_text());r=next(x for x in r['logs'] if x['step']==step);rows.append(r);group.append(r['test']);seed_results.setdefault(str(seed),[]).append(r['test'])
   for g,v in r['test_games'].items():by.setdefault(g,[]).append(v)
  folds[str(f)]={k:float(np.mean([d[k] for d in group])) for k in group[0]}
 a=np.array([np.mean(v,axis=0) for v in by.values()]);idx=np.random.default_rng(111777).integers(len(a),size=(4000,len(a)));improve=np.stack((a[:,1]-a[:,0],a[:,3]-a[:,2]),1);ci=np.quantile(improve[idx].mean(1),[.025,.975],axis=0)
 avg=dict(zip(['mse','zero_mse','regret','zero_regret'],a.mean(0).tolist()));perseed={s:{k:float(np.mean([d[k] for d in ds])) for k in ds[0]} for s,ds in seed_results.items()}
 gate=dict(all_folds_5percent=all(v['mse']<.95*v['zero_mse'] and v['regret']<.95*v['zero_regret'] for v in folds.values()),each_seed_regret=all(v['regret']<v['zero_regret'] for v in perseed.values()),game_ci_positive=bool((ci[0]>0).all()))
 return dict(size=size,condition=condition,step=step,mean=avg,folds=folds,update_seeds=perseed,game_clusters=len(a),mse_improvement_ci95=ci[:,0].tolist(),regret_improvement_ci95=ci[:,1].tolist(),gate=gate,advance=all(gate.values()),train={k:float(np.mean([r['train'][k] for r in rows])) for k in rows[0]['train']},note='All seeds averaged within each game. Fixed test folds disjoint, training folds overlap; seed spread is separate from game-cluster CI.')

def run(size):
 results=[]
 for f in range(3):
  for seed in SEEDS:
   for condition in ['plain','decay']:
    path=ROOT/f'n{size}-f{f}-s{seed}-{condition}/result.json'
    if path.exists():r=json.loads(path.read_text())
    else:
     spent=sum(json.loads(p.read_text())['seconds'] for p in ROOT.glob('n*/result.json'));remaining=1800-spent
     if remaining<1:raise RuntimeError('global training time cap reached')
     r=train(size,f,seed,condition,time_cap=min(40,remaining))
    assert r['step']==3000,'run stopped at per-model cap; do not treat as completed'
    results.append(r)
 report=[aggregate(size,c,step) for c in ['plain','decay'] for step in [1000,3000]];write(ROOT/f'curve-{size}.json',report)
 if size==960:
  small=json.loads((ROOT/'curve-240.json').read_text());conditions={}
  for c in ['plain','decay']:
   a=next(x['mean'] for x in small if x['condition']==c and x['step']==1000);b=next(x['mean'] for x in report if x['condition']==c and x['step']==1000)
   conditions[c]=bool(b['mse']<=a['mse'] and b['regret']<=a['regret'] and (b['mse']<=.95*a['mse'] or b['regret']<=.95*a['regret']))
  write(ROOT/'extension-gate.json',dict(extend=any(conditions.values()),conditions=conditions,criterion='fixed bank, 1000 updates: both metrics nonworse and at least one improves 5%'))
 accepted=[r['condition'] for r in report if r['step']==1000 and r['advance']]
 final=size==2400 or (size==960 and not json.loads((ROOT/'extension-gate.json').read_text())['extend'])
 write(ROOT/'decision.json',dict(size=size,accepted_conditions=accepted,advance=bool(accepted),final=final,criterion='registered 1000-update heldout teacher-value gate only; no battle override'))
 total=sum(json.loads(p.read_text())['seconds'] for p in ROOT.glob('n*/result.json'));assert total<1800
 write(ROOT/'training-time.json',dict(seconds=total,cap=1800));print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--size',type=int,required=True);run(p.parse_args().size)
