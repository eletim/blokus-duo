"""Fixed endpoints: no checkpoint or seed selection from validation/holdout."""
import argparse,json
from pathlib import Path
import numpy as np
from . import ppo_gpu_eval as e
from .ppo_minibatch import R,SEEDS,write
from .ppo_minibatch_audit import CACHE
from .ai import seed_all
from .ppo_gpu_train import sha
e.R=R;e.CACHE=CACHE
def model(mb,seed,step):return R/f'batch-{mb}-seed-{seed}'/f'inference-{step}.pt'
def base(seed):return Path(f'artifacts/ppo-four-gpu/seed-{seed}/inference-4096.pt')
def gen2(seed):return Path(f'artifacts/ppo-gen2/seed-{seed-1750000}.pt')
def validation():
 seed_all(1764000);prefixes=e.bank('validation',64,17641000);end=json.loads((R/'training-complete.json').read_text())['common_iteration'];curves={}
 for step in [s for s in (4608,5120) if s<=end]:
  groups={label:[] for label in ('B-A','A-gen2','B-gen2','A-base','B-base')}
  for seed in SEEDS:
   for label,left,right in [('B-A',model(64,seed,step),model(16,seed,step)),('A-gen2',model(16,seed,step),gen2(seed)),('B-gen2',model(64,seed,step),gen2(seed)),('A-base',model(16,seed,step),base(seed)),('B-base',model(64,seed,step),base(seed))]:
    result=e.match(left,right,prefixes,R/f'validation-{step}-{seed}-{label}.json');groups[label].append((seed,result));print('VALIDATION',step,seed,label,result['summary']['point_rate'],flush=True)
  curves[str(step)]={label:e.cluster(matches) for label,matches in groups.items()}
 final=curves[str(end)];gate=final['B-A']['point']>=.5 and final['B-gen2']['point']>=final['A-gen2']['point']
 write(R/'selection.json',dict(fixed_iteration=end,no_selection_search=True,gate=bool(gate),curves=curves,checkpoints={f'{mb}-{seed}':dict(path=str(model(mb,seed,end)),sha256=sha(model(mb,seed,end))) for mb in (16,64) for seed in SEEDS}));print('GATE',gate,flush=True)
def holdout():
 seed_all(1766000);lock=json.loads((R/'selection.json').read_text());assert lock['gate'];step=lock['fixed_iteration']
 for d in lock['checkpoints'].values():assert sha(d['path'])==d['sha256']
 prefixes=e.bank('holdout',96,17661000);groups={k:[] for k in ('B-A','B-gen2','B-base')}
 for seed in SEEDS:
  for label,right in [('B-A',model(16,seed,step)),('B-gen2',gen2(seed)),('B-base',base(seed))]:
   result=e.match(model(64,seed,step),right,prefixes,R/f'holdout-{seed}-{label}.json');groups[label].append((seed,result));print('HOLDOUT',seed,label,result['summary']['point_rate'],flush=True)
 write(R/'holdout-summary.json',{label:e.cluster(matches) for label,matches in groups.items()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','holdout','audit']);a=p.parse_args();{'validation':validation,'holdout':holdout,'audit':e.audit_eval}[a.mode]()
