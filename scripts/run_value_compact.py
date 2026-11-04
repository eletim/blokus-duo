"""Revision 3: compact regularized value, more games, batched-root search."""
import json
import shutil
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.ai import seed_all
from blokus.research import sha,source_hashes
from blokus.tournament import make_prefixes,read_prefixes
from blokus.value_data import collect
from blokus.value_train import train
from blokus.value_experiment import timed_match,evaluate,write


def main():
    root=Path('artifacts/value-search/compact');root.mkdir(parents=True,exist_ok=True)
    if (root/'run-config.json').exists():
        if '--resume-data' not in sys.argv or (root/'seed-201/latest.pt').exists():raise ValueError('existing compact experiment')
        shutil.copy2(root/'run-config.json',root/'failed-run-config.json')
        expected=json.loads((root/'expanded.json').read_text())['sha256']
        if sha(root/'expanded.npz')!=expected:raise ValueError('dataset hash mismatch')
    begin=time.perf_counter();seed_all(201)
    prior=Path('artifacts/value-search/run');validation=read_prefixes(prior/'validation-prefixes.json')
    write(root/'run-config.json',dict(source_hashes=source_hashes(),arch='pooled',augment='transpose and 180-degree rotation',
          train_seeds=[201,202,203],expanded_games=6000,expanded_seed=130000,holdout_seed=94000,budget_ms=6,scheduler='batched',
          steps=4000,training_seconds_limit_per_seed=120,selection='mean score difference against static and original C under same new batched scheduler'))
    if not (root/'expanded.npz').exists():collect(root/'expanded.npz',games=6000,seed=130000,workers=6)
    datasets=[prior/'games.npz',prior/'branches.npz',root/'expanded.npz'];training=[];selected={};records=[]
    for seed in [201,202,203]:
        folder=root/f'seed-{seed}'
        summary=train(datasets,folder,seed=seed,steps=4000,seconds=120,arch='pooled',augment=True,branch_weight=8)
        training.append(summary);choices=[]
        for alpha in [.25,.5,1.]:
            spec=dict(kind='value',checkpoint=str(folder/'predictive-best.pt'),alpha=alpha,gate=24,scheduler='batched');scores=[]
            for opponent in ['static','C']:
                result=timed_match(spec,dict(kind=opponent,scheduler='batched'),validation)
                result.update(seed=seed,alpha=alpha);records.append(result);scores.append(result['summary']['mean_difference'])
                write(root/'validation-results.json',records)
            choices.append(dict(spec=spec,criterion=sum(scores)/2))
        chosen=max(choices,key=lambda x:x['criterion']);shutil.copy2(folder/'predictive-best.pt',folder/'selected.pt')
        chosen['spec']['checkpoint']=str(folder/'selected.pt');selected[seed]=dict(**chosen,checkpoint_sha256=sha(folder/'selected.pt'),choices=choices)
        with open('PROGRESS.md','a') as f:f.write(f"\nCompact stage3 seed{seed}: {summary['steps']}更新、{summary['training_seconds']:.2f}秒、best predictive MSE={summary['best_mse']:.5f}、validation criterion={chosen['criterion']:+.3f}。\n")
        print('COMPACT SELECT',seed,chosen,flush=True)
    write(root/'selection.json',dict(selected=selected,validation_sha256=validation['sha256'],source_hashes=source_hashes()))
    holdout=make_prefixes(94000,24);write(root/'holdout-prefixes.json',holdout)
    specs={f'value-{seed}':v['spec'] for seed,v in selected.items()};specs.update(static=dict(kind='static',scheduler='batched'),C=dict(kind='C',scheduler='batched'))
    pairs=[(f'value-{s}',b) for s in [201,202,203] for b in ['static','C']]+[('C','static')]
    evaluate(specs,pairs,holdout,root/'holdout-results.json')
    write(root/'run-summary.json',dict(training=training,gpu_training_seconds=sum(t['invocation_seconds'] for t in training),wall_seconds=time.perf_counter()-begin,completed=True))
    print('COMPACT COMPLETE',flush=True)
if __name__=='__main__':main()
