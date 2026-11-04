"""Reproducible grouped-return training and new-prefix paired comparisons."""
import argparse
import json
import shutil
import time
from pathlib import Path
from .ai import seed_all
from .research import source_hashes,sha
from .value_experiment import write,timed_match,evaluate
from .tournament import make_prefixes
from .ranking_data import collect
from .ranking_train import train

BASE=dict(kind='value',checkpoint='artifacts/value-search/compact/seed-201/predictive-best.pt',alpha=.5,gate=24)


def run(output,games=1200,steps=6000,seconds=180):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise ValueError('existing experiment')
    begin=time.perf_counter();seed_all(301)
    write(root/'config.json',dict(source_hashes=source_hashes(),seeds=[301,302,303],train_seed=200000,validation_seed=210000,validation_prefixes=32,
          holdout_seed=220000,holdout_prefixes=96,steps=steps,gpu_seconds_limit_per_seed=seconds,budget_ms=6,baseline=BASE,
          selection='minimum held-out-group pair MSE checkpoint, then maximum validation mean score vs static and prior-value among alpha .25/.5/1'))
    data=collect(root/'groups.npz',games=games)
    validation=make_prefixes(210000,32);write(root/'validation-prefixes.json',validation)
    selected={};records=[];summaries=[]
    for seed in [301,302,303]:
        folder=root/f'seed-{seed}';summaries.append(train(root/'groups.npz',folder,seed,steps,seconds))
        choices=[]
        for alpha in [.25,.5,1.]:
            spec=dict(kind='value',checkpoint=str(folder/'predictive-best.pt'),alpha=alpha,gate=24);scores=[]
            for name,opponent in [('static',dict(kind='static')),('prior-value',BASE)]:
                r=timed_match(spec,opponent,validation);r.update(seed=seed,alpha=alpha,opponent=name);records.append(r)
                scores.append(r['summary']['mean_difference']);write(root/'validation-results.json',records)
            choices.append(dict(spec=spec,criterion=sum(scores)/2))
        chosen=max(choices,key=lambda x:x['criterion']);shutil.copy2(folder/'predictive-best.pt',folder/'selected.pt');chosen['spec']['checkpoint']=str(folder/'selected.pt')
        selected[seed]=dict(**chosen,checkpoint_sha256=sha(folder/'selected.pt'),choices=choices)
        with open('PROGRESS.md','a') as f:f.write(f"\nRanking seed{seed}: {summaries[-1]['invocation_seconds']:.2f}秒/{summaries[-1]['steps']}更新、validation criterion={chosen['criterion']:+.3f}, alpha={chosen['spec']['alpha']}。\n")
        print('SELECT',seed,chosen,flush=True)
    write(root/'selection.json',dict(selected=selected,validation_sha256=validation['sha256']))
    # Created only after all selection decisions have been durably saved.
    holdout=make_prefixes(220000,96);write(root/'holdout-prefixes.json',holdout)
    specs={f'ranking-{s}':v['spec'] for s,v in selected.items()};specs.update(static=dict(kind='static'),prior=BASE)
    write(root/'play-specs.json',specs)
    pairs=[(f'ranking-{s}',b) for s in selected for b in ['static','prior']]+[('prior','static')]
    evaluate(specs,pairs,holdout,root/'holdout-results.json')
    write(root/'summary.json',dict(completed=True,wall_seconds=time.perf_counter()-begin,data=data,training=summaries,
          gpu_training_seconds=sum(x['invocation_seconds'] for x in summaries),holdout_games=96*2*len(pairs),validation_games=32*2*len(records)))
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','collect','train']);p.add_argument('--output',default='artifacts/paired-ranking')
    p.add_argument('--games',type=int,default=1200);p.add_argument('--steps',type=int,default=6000);p.add_argument('--seconds',type=float,default=180)
    p.add_argument('--seed',type=int,default=301);p.add_argument('--data');p.add_argument('--device',default='cuda');p.add_argument('--resume')
    a=p.parse_args()
    if a.command=='run':run(a.output,a.games,a.steps,a.seconds)
    elif a.command=='collect':collect(a.output,a.games,a.seed)
    else:train(a.data,a.output,a.seed,a.steps,a.seconds,a.device,a.resume)
if __name__=='__main__':main()
