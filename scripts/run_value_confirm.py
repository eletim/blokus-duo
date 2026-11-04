"""Revision 4: retain compact training, restore stable incremental time scheduler.

No new model training or holdout selection; alpha chosen again on validation only.
"""
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.value_experiment import write,timed_match,evaluate
from blokus.tournament import read_prefixes,make_prefixes
from blokus.research import sha,source_hashes
from blokus.ai import seed_all

root=Path('artifacts/value-search/confirm');root.mkdir(parents=True,exist_ok=True)
if (root/'run-config.json').exists():raise ValueError('existing confirmation experiment')
seed_all(201);begin=time.perf_counter()
write(root/'run-config.json',dict(source_hashes=source_hashes(),reason='batched scheduler has excessive overrun/fallback rate; restore incremental completion',budget_ms=6,holdout_seed=95000,training='reuse fixed compact models; no holdout weight selection'))
validation=read_prefixes('artifacts/value-search/run/validation-prefixes.json');selected={};records=[]
for seed in [201,202,203]:
    checkpoint=f'artifacts/value-search/compact/seed-{seed}/predictive-best.pt';choices=[]
    for alpha in [.25,.5,1.]:
        spec=dict(kind='value',checkpoint=checkpoint,alpha=alpha,gate=24);scores=[]
        for opponent in ['static','C']:
            result=timed_match(spec,dict(kind=opponent),validation);result.update(seed=seed,alpha=alpha)
            records.append(result);scores.append(result['summary']['mean_difference']);write(root/'validation-results.json',records)
        choices.append(dict(spec=spec,criterion=sum(scores)/2))
    chosen=max(choices,key=lambda x:x['criterion']);selected[seed]=dict(**chosen,checkpoint_sha256=sha(checkpoint),choices=choices)
    print('SELECT',seed,chosen,flush=True)
write(root/'selection.json',dict(selected=selected,validation_sha256=validation['sha256']))
holdout=make_prefixes(95000,24);write(root/'holdout-prefixes.json',holdout)
specs={f'value-{s}':v['spec'] for s,v in selected.items()};specs.update(static=dict(kind='static'),C=dict(kind='C'))
write(root/'play-specs.json',specs)
evaluate(specs,[(f'value-{s}',b) for s in [201,202,203] for b in ['static','C']]+[('C','static')],holdout,root/'holdout-results.json')
write(root/'run-summary.json',dict(completed=True,wall_seconds=time.perf_counter()-begin,new_gpu_training_seconds=0))
print('CONFIRM COMPLETE',flush=True)
