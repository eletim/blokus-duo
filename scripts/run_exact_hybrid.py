"""Mobility-gated full-search follow-up, preregistered before first holdout."""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.deep_experiment import tournament
from blokus.ai import seed_all
from blokus.aligned_data import BASE
from blokus.research import sha,source_hashes
from blokus.value_experiment import write
from blokus.tournament import make_prefixes

CONFIGS={f'mobility{n}':dict(root_width=12,width=3,deep_gate=24,exact_pieces=0,exact_mobility=n,only_exact=True,max_depth=2) for n in [6,10]}


def main():
    root=Path('artifacts/exact-hybrid');root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise ValueError('existing run')
    seed_all(702);start=time.perf_counter();write(root/'config.json',dict(configs=CONFIGS,checkpoint=BASE,checkpoint_sha256=sha(BASE),source_hashes=source_hashes(),dev_seed=511000,validation_seed=521000,holdout_seed=531000,dev_prefixes=32,validation_prefixes=32,holdout_prefixes=96,training=False))
    dev=make_prefixes(511000,32);write(root/'dev-prefixes.json',dev)
    dr=tournament([(f'{n}-{ms}',dict(kind='deep',budget_ms=ms,**cfg),dict(kind='old',budget_ms=ms)) for n,cfg in CONFIGS.items() for ms in [25,100]],dev,root/'dev-results.json')
    best=max(CONFIGS,key=lambda n:sum(r['summary']['mean_difference'] for r in dr['results'] if r['name'].startswith(n+'-'))/2)
    write(root/'dev-selection.json',dict(config=best,criterion='mean same-budget point difference over25/100ms',frozen_config=CONFIGS[best]))
    small=make_prefixes(511999,8);write(root/'six-prefixes.json',small);tournament([(best+'-6',dict(kind='deep',budget_ms=6,**CONFIGS[best]),dict(kind='old',budget_ms=6))],small,root/'six-results.json')
    validation=make_prefixes(521000,32);write(root/'validation-prefixes.json',validation)
    vr=tournament([(f'{best}-{ms}',dict(kind='deep',budget_ms=ms,**CONFIGS[best]),dict(kind='old',budget_ms=ms)) for ms in [25,100]],validation,root/'validation-results.json')
    choice=max(vr['results'],key=lambda r:r['summary']['mean_difference']);spec=choice['left']['spec'];write(root/'selection.json',dict(spec=spec,criterion='newvalidation same-budget mean point difference',validation_mean=choice['summary']['mean_difference'],validation_prefix_sha256=validation['sha256']))
    earlier=json.loads(Path('artifacts/deep-search/selection.json').read_text())['spec'];control=dict(earlier,budget_ms=spec['budget_ms'])
    hold=make_prefixes(531000,96);write(root/'holdout-prefixes.json',hold);write(root/'control-spec.json',control)
    with open('PROGRESS.md','a') as f:f.write(f'\nExact-hybrid fixed aftervalidation: {spec}; fresh531000 begins, earlier method same-budget control={control}.\n')
    tournament([('selected-vs-old',spec,dict(kind='old',budget_ms=spec['budget_ms'])),('selected-vs-earlier',spec,control)],hold,root/'holdout-results.json')
    write(root/'summary.json',dict(completed=True,seconds=time.perf_counter()-start,selected=spec,training_updates=0,checkpoint_sha256=sha(BASE)));print('COMPLETE',flush=True)
if __name__=='__main__':main()
