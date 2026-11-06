"""LR contrast with predeclared progression gate and fixed endpoint."""
import argparse,json
from pathlib import Path
from . import ppo_gpu_eval as e
from .ppo_lr64 import R,SEEDS,write
from .ppo_lr64_audit import CACHE
from .ai import seed_all
from .ppo_gpu_train import sha
e.R=R;e.CACHE=CACHE
def model(condition,seed,step):return R/f'{condition}-seed-{seed}'/f'inference-{step}.pt'
def gen2(seed):return Path(f'artifacts/ppo-gen2/seed-{seed-1750000}.pt')
def validation():
 assert json.loads((R/'training-audit.json').read_text())['pass_']
 seed_all(1774000);prefixes=e.bank('validation',64,17741000);end=json.loads((R/'training-complete.json').read_text())['common_iteration'];curves={}
 for step in [s for s in (4608,5120) if s<=end]:
  groups={label:[] for label in ('C-B','C-A','A-gen2','B-gen2','C-gen2')}
  for seed in SEEDS:
   for label,left,right in [('C-B',model('C',seed,step),model('B',seed,step)),('C-A',model('C',seed,step),model('A',seed,step))]+[(c+'-gen2',model(c,seed,step),gen2(seed)) for c in 'ABC']:
    result=e.match(left,right,prefixes,R/f'validation-{step}-{seed}-{label}.json');groups[label].append((seed,result));print('VALIDATION',step,seed,label,result['summary']['point_rate'],flush=True)
  curves[str(step)]={label:e.cluster(matches) for label,matches in groups.items()}
 final=curves[str(end)];gate=final['C-gen2']['point']>.5 and final['C-gen2']['difference']>=0 and final['C-B']['point_ci95'][0]>.45
 write(R/'selection.json',dict(fixed_iteration=end,no_selection_search=True,gate=bool(gate),curves=curves,checkpoints={f'{c}-{seed}':dict(path=str(model(c,seed,end)),sha256=sha(model(c,seed,end))) for c in 'ABC' for seed in SEEDS}));print('GATE',gate,flush=True)
def holdout():
 audit=json.loads((R/'evaluation-audit.json').read_text());assert audit['pass_'] and all(r['start_overlap']==0 and r['later_state_overlap']==0 for r in audit['rows'])
 seed_all(1776000);lock=json.loads((R/'selection.json').read_text());assert lock['gate'];step=lock['fixed_iteration']
 for d in lock['checkpoints'].values():assert sha(d['path'])==d['sha256']
 prefixes=e.bank('holdout',96,17761000);groups={k:[] for k in ('C-B','C-gen2','C-A')}
 for seed in SEEDS:
  for label,right in [('C-B',model('B',seed,step)),('C-gen2',gen2(seed)),('C-A',model('A',seed,step))]:
   result=e.match(model('C',seed,step),right,prefixes,R/f'holdout-{seed}-{label}.json');groups[label].append((seed,result));print('HOLDOUT',seed,label,result['summary']['point_rate'],flush=True)
 write(R/'holdout-summary.json',{label:e.cluster(matches) for label,matches in groups.items()})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['validation','holdout','audit']);a=p.parse_args();{'validation':validation,'holdout':holdout,'audit':e.audit_eval}[a.mode]()
