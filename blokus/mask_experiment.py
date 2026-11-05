"""Finite three-fold/three-seed grid, registered extension and game diagnostic gate."""
import argparse,json,time
import numpy as np
from .mask_data import ROOT
from .mask_train import train
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

def run(size=2400):
 for f in range(3):
  for seed in SEEDS:
   for condition in ['summary','cells']:
    path=ROOT/f'n{size}-f{f}-s{seed}-{condition}/result.json'
    spent=sum(json.loads(p.read_text())['seconds'] for p in ROOT.glob('n*/result.json'))
    if path.exists():r=json.loads(path.read_text())
    else:
     assert spent<1200
     r=train(size,f,seed,condition,time_cap=min(120,1200-spent))
    assert r['step']==1000,'incomplete training cannot pass gate'
 report=[aggregate(size,c) for c in ['summary','cells']]
 write(ROOT/'comparison.json',report)
 write(ROOT/'decision.json',dict(advance=report[1]['advance'],accepted_conditions=[r['condition'] for r in report if r['advance']],final=True,criterion='unchanged registered generalization gate; spatial model must pass'))
 write(ROOT/'training-time.json',dict(seconds=sum(json.loads(p.read_text())['seconds'] for p in ROOT.glob('n*/result.json')),cap=1200))
 print(json.dumps(report),flush=True)
if __name__=='__main__':run()
