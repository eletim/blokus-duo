"""Shared-data loss ablations; select one method before a fresh holdout."""
import argparse
import json
import time
from pathlib import Path
from .ai import seed_all
from .research import source_hashes,sha
from .value_experiment import write,timed_match,evaluate
from .tournament import make_prefixes
from .aligned_data import collect,BASE
from .aligned_train import train

PRIOR=dict(kind='value',checkpoint=BASE,alpha=.5,gate=24)


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise ValueError('existing experiment')
    quality=json.loads((root/'pilot.quality.json').read_text());assert json.loads((root/'pilot.audit.json').read_text())['status']=='PASS'
    repeats=quality['selected_training_repeats'];seed_all(501);start=time.perf_counter()
    def limit():
        if time.perf_counter()-start>6900:
            write(root/'partial.json',dict(reason='wall budget approaching 2h',elapsed=time.perf_counter()-start));raise TimeoutError('wall budget')
    config=dict(source_hashes=source_hashes(),seeds=[501,502,503],modes=['absolute','difference','ranking'],train_seed=300000,games=600,groups_per_game=2,
                repeats=repeats,pilot_sha256=sha(root/'pilot.npz'),validation_seed=310000,validation_prefixes=32,holdout_seed=320000,holdout_prefixes=96,
                steps_per_model=6000,gpu_seconds_limit_per_model=180,total_main_training_limit=1620,budget_ms=6,alpha=.5,gate=24,
                selection='highest mean validation score difference across all three seeds against static and prior; holdout only winning mode')
    write(root/'config.json',config)
    data=collect(root/'groups.npz',games=600,seed=300000,groups=2,repeats=repeats)
    validation=make_prefixes(310000,32);write(root/'validation-prefixes.json',validation)
    records=[];summaries=[];choices=[]
    for mode in config['modes']:
        scores=[];specs={}
        for seed in config['seeds']:
            limit();folder=root/mode/f'seed-{seed}';summary=train(root/'groups.npz',folder,mode,seed=seed);summaries.append(summary)
            spec=dict(kind='value',checkpoint=str(folder/'selected.pt'),alpha=.5,gate=24);specs[seed]=spec
            for name,opponent in [('static',dict(kind='static')),('prior',PRIOR)]:
                limit();r=timed_match(spec,opponent,validation);r.update(mode=mode,seed=seed,opponent=name);records.append(r)
                scores.append(r['summary']['mean_difference']);write(root/'validation-results.json',records)
            with open('PROGRESS.md','a') as f:f.write(f"\nAligned {mode}/{seed}: {summary['steps']}更新/{summary['invocation_seconds']:.2f}秒、best decision regret={summary['best'][0]:.3f}点、validation直近2相手mean={sum(scores[-2:])/2:+.3f}。\n")
            print('VALIDATED',mode,seed,scores[-2:],flush=True)
        choices.append(dict(mode=mode,criterion=sum(scores)/len(scores),specs=specs))
    chosen=max(choices,key=lambda x:x['criterion'])
    write(root/'selection.json',dict(chosen=chosen,choices=choices,validation_sha256=validation['sha256'],checkpoint_hashes={s:sha(v['checkpoint']) for s,v in chosen['specs'].items()}))
    print('SELECT',chosen,flush=True)
    limit();holdout=make_prefixes(320000,96);write(root/'holdout-prefixes.json',holdout)
    specs={f'aligned-{s}':spec for s,spec in chosen['specs'].items()};specs.update(static=dict(kind='static'),prior=PRIOR);write(root/'play-specs.json',specs)
    pairs=[(f'aligned-{s}',other) for s in config['seeds'] for other in ['static','prior']]+[('prior','static')]
    evaluate(specs,pairs,holdout,root/'holdout-results.json')
    write(root/'summary.json',dict(completed=True,wall_seconds=time.perf_counter()-start,data=data,training=summaries,gpu_training_seconds=sum(t['invocation_seconds'] for t in summaries),holdout_games=1344,validation_games=1152,selected_mode=chosen['mode']))
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','train']);p.add_argument('--output',default='artifacts/search-aligned');p.add_argument('--data')
    p.add_argument('--mode',choices=['absolute','difference','ranking'],default='absolute');p.add_argument('--seed',type=int,default=501)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--seconds',type=float,default=180);p.add_argument('--device',default='cuda');p.add_argument('--resume')
    a=p.parse_args()
    if a.command=='run':run(a.output)
    else:train(a.data,a.output,a.mode,a.seed,a.steps,a.seconds,a.device,a.resume)
if __name__=='__main__':main()
