"""Preregistered pilot gate, compute-matched diversity comparison and selector study."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import torch
from .ai import seed_all
from .aligned_data import BASE
from .value_search import OutcomeNet
from .research import sha,source_hashes
from .value_experiment import write
from .tournament import make_prefixes
from .visited_data import gather,label,stratified_subset
from .visited_train import train,load,metrics
from .visited_eval import match,tournament

PRIOR=dict(kind='value',checkpoint=BASE,alpha=.5,gate=24)
SEEDS=[601,602,603]


def weight_hash(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);h=hashlib.sha256()
    for k,v in sorted(ck['model'].items()):h.update(k.encode());h.update(v.contiguous().numpy().tobytes())
    return h.hexdigest()


def stage(root,name,validation_seed,n_prefix,steps,seconds):
    folder=root/name;fixture=make_prefixes(validation_seed,n_prefix);write(folder/'validation-prefixes.json',fixture)
    records=[];summaries=[];choices=[];cache={}
    for condition in ['wide','dense']:
        for seed in SEEDS:
            out=folder/condition/f'seed-{seed}';summaries.append(train(folder/f'{condition}.npz',root/'dev.npz',out,seed,steps,seconds))
            for selector in ['raw','live']:
                spec=dict(kind='value',checkpoint=str(out/f'{selector}.pt'),alpha=.5,gate=24);fingerprint=weight_hash(spec['checkpoint'])
                for opponent,other in [('static',dict(kind='static')),('prior',PRIOR)]:
                    key=(fingerprint,opponent)
                    if key in cache:
                        r=copy.deepcopy(cache[key]);r['left']['spec']=spec;r['left']['checkpoint_sha256']=sha(spec['checkpoint']);r['reused_identical_weights']=True
                    else:
                        r=match(spec,other,fixture);cache[key]=copy.deepcopy(r);r['reused_identical_weights']=False
                    r.update(condition=condition,selector=selector,seed=seed,opponent=opponent,weight_sha256=fingerprint);records.append(r);write(folder/'validation-results.json',records)
            with open('PROGRESS.md','a') as f:f.write(f"\nVisited {name}/{condition}/{seed}: {summaries[-1]['invocation_steps']}更新/{summaries[-1]['invocation_seconds']:.2f}秒、dev best raw/live={summaries[-1]['best']}。\n")
            print('VALIDATED',name,condition,seed,flush=True)
        for selector in ['raw','live']:
            values=[r['summary']['mean_difference'] for r in records if r['condition']==condition and r['selector']==selector]
            choices.append(dict(condition=condition,selector=selector,criterion=sum(values)/len(values)))
    chosen=max(choices,key=lambda x:x['criterion']);write(folder/'selection.json',dict(chosen=chosen,choices=choices,validation_sha256=fixture['sha256']))
    write(folder/'summary.json',dict(training=summaries,gpu_training_seconds=sum(x['invocation_seconds'] for x in summaries),actual_validation_games=sum(len(r['games']) for r in records if not r['reused_identical_weights']),reported_validation_contrasts=len(records)))
    return chosen,records,summaries


def run(root):
    root=Path(root);prepare=json.loads((root/'prepare-summary.json').read_text())
    if (root/'run-config.json').exists():raise ValueError('existing run')
    start=time.perf_counter();seed_all(601)
    write(root/'run-config.json',dict(source_hashes=source_hashes(),seeds=SEEDS,loss='absolute',pilot_steps=3000,pilot_seconds_per_model=90,full_steps=6000,full_seconds_per_model=120,
        pilot_validation_seed=410000,full_validation_seed=411000,holdout_seed=450000,holdout_prefixes=96,alpha=.5,gate=24,
        expansion_gate=dict(relative_dev_live_regret_improvement=.1,validation_live_vs_prior=.25),full_label_wall_target=2100,max_wide_roots=10000,
        identical_weights_validation='reuse same match and mark; do not invent selector difference from timing noise'))
    model=OutcomeNet('pooled').cuda();model.load_state_dict(torch.load(BASE,map_location='cpu',weights_only=False)['model']);initial=metrics(model,load(root/'dev.npz','cuda'));write(root/'initial-dev.json',initial);del model
    chosen,records,summaries=stage(root,'pilot',410000,16,3000,90)
    gates={}
    for condition in ['wide','dense']:
        # Summaries are wide seeds, then dense seeds in a stable order.
        subset=summaries[:3] if condition=='wide' else summaries[3:]
        regret=sum(x['best']['live'][0] for x in subset)/3
        prior=sum(r['summary']['mean_difference'] for r in records if r['condition']==condition and r['selector']=='live' and r['opponent']=='prior')/3
        baseline=initial['live']['decision_regret_points'];improvement=1-regret/baseline if baseline>0 else 0
        gates[condition]=dict(dev_live_regret=regret,relative_improvement=improvement,validation_vs_prior=prior,expand=improvement>=.1 and prior>=.25)
    expand=any(g['expand'] for g in gates.values());write(root/'gate.json',dict(initial=initial,gates=gates,expand=expand))
    final_stage='pilot'
    if expand:
        cost=(prepare['labels']['wide']['seconds']+prepare['labels']['dense']['seconds'])/512
        elapsed=time.time()-Path('/tmp/diversity-request.json').stat().st_mtime if Path('/tmp/diversity-request.json').exists() else time.perf_counter()-start+prepare['seconds']
        label_budget=min(2100,max(0,6900-elapsed-900))
        n=max(512,min(10000,int(label_budget/cost)//4*4));full=root/'full';full.mkdir(exist_ok=True)
        write(full/'plan.json',dict(wide_roots=n,dense_roots=n//4,repeats=[4,16],estimated_label_seconds=n*cost,reason='pilot signal gate passed'))
        visited=gather(full/'visited.json',n,440000)
        label(full/'wide.npz',visited['records'],4);label(full/'dense.npz',stratified_subset(visited['records'],n//4),16)
        chosen,records,summaries=stage(root,'full',411000,32,6000,120);final_stage='full'
    write(root/'final-selection.json',dict(stage=final_stage,chosen=chosen,expand=expand))
    holdout=make_prefixes(450000,96);write(root/'holdout-prefixes.json',holdout)
    specs={f'visited-{s}':dict(kind='value',checkpoint=str(root/final_stage/chosen['condition']/f'seed-{s}'/f"{chosen['selector']}.pt"),alpha=.5,gate=24) for s in SEEDS}
    specs.update(static=dict(kind='static'),prior=PRIOR);write(root/'play-specs.json',specs)
    tournament(specs,[(f'visited-{s}',o) for s in SEEDS for o in ['static','prior']]+[('prior','static')],holdout,root/'holdout-results.json')
    write(root/'summary.json',dict(completed=True,run_seconds=time.perf_counter()-start,prepare_seconds=prepare['seconds'],expanded=expand,final_stage=final_stage,chosen=chosen,
        gpu_training_seconds=sum(json.loads((root/s/'summary.json').read_text())['gpu_training_seconds'] for s in ['pilot','full'] if (root/s/'summary.json').exists())))
    print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','train']);p.add_argument('--output',default='artifacts/visited-diversity');p.add_argument('--data');p.add_argument('--dev');p.add_argument('--seed',type=int,default=601);p.add_argument('--steps',type=int,default=3000);p.add_argument('--seconds',type=float,default=90);p.add_argument('--device',default='cuda');p.add_argument('--resume')
    a=p.parse_args()
    if a.command=='run':run(a.output)
    else:train(a.data,a.dev,a.output,a.seed,a.steps,a.seconds,a.device,a.resume)
if __name__=='__main__':main()
