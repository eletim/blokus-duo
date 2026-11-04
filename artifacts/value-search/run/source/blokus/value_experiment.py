"""Reproducible profile/train/timed-tournament/run/play CLI for outcome search."""
import argparse
import json
import random
import shutil
import time
from pathlib import Path
import numpy as np
from .ai import seed_all
from .engine import State,legal_moves,play
from .research import sha,source_hashes
from .tournament import make_prefixes,read_prefixes,prefix_state,summarize
from .value_search import BudgetAgent
from .value_data import collect
from .value_train import train


def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2))


def timed_match(left,right,prefixes,budget=6,device='cuda'):
    agents=[BudgetAgent(**spec,budget_ms=budget,device=device) for spec in (left,right)]
    games=[];start=time.perf_counter()
    for i,prefix in enumerate(prefixes['prefixes']):
        for seat in (0,1):
            s=prefix_state(prefix);rngs=[random.Random(712+i*2+p) for p in (0,1)];turns=0;timings=[]
            while not s.done:
                who=0 if s.turn==seat else 1;agent=agents[who]
                move=agent.act(s,rngs[s.turn]);timings.append([who,agent.times[-1]*1000])
                if move is None:
                    assert not legal_moves(s);play(s,None,validate=False)
                else:play(s,move)
                turns+=1;assert turns<=86
            games.append(dict(prefix_id=prefix['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),
                              plies_after_prefix=turns,decision_ms=timings))
    def metadata(spec,agent):
        result=dict(spec=spec,stats=agent.stats())
        if spec.get('checkpoint'):result['checkpoint_sha256']=sha(spec['checkpoint'])
        return result
    return dict(left=metadata(left,agents[0]),right=metadata(right,agents[1]),prefix_sha256=prefixes['sha256'],
                summary=summarize(games),games=games,wall_seconds=time.perf_counter()-start,
                timing_protocol='same wall allowance includes generation/ranking/GPU; complete child only; late child discarded; no padding',budget_ms=budget)


def evaluate(specs,pairs,prefixes,path,budget=6,device='cuda'):
    report=dict(prefix_sha256=prefixes['sha256'],source_hashes=source_hashes(),results=[])
    for left,right in pairs:
        result=timed_match(specs[left],specs[right],prefixes,budget,device);result['names']=[left,right]
        report['results'].append(result);write(path,report)
        print(json.dumps(dict(pair=[left,right],**result['summary'])),flush=True)
    return report


def stage(root,datasets,seeds,prefixes,stage_id,initial=None,steps=1600,gate=42):
    results=[];selected={};training=[]
    for seed in seeds:
        folder=root/f'seed-{seed}'
        summary=train(datasets,folder,seed=seed,steps=steps,seconds=120,initialize=initial[seed] if initial else None,branch_weight=8 if stage_id==2 else 1)
        training.append(summary)
        candidates=[]
        for alpha in [.25,.5,1.]:
            spec=dict(kind='value',checkpoint=str(folder/'predictive-best.pt'),alpha=alpha,gate=gate)
            scores=[]
            for opponent in ('static','C'):
                result=timed_match(spec,dict(kind=opponent),prefixes)
                result.update(seed=seed,alpha=alpha,stage=stage_id);results.append(result)
                scores.append(result['summary']['mean_difference']);write(root/'validation-results.json',results)
            candidates.append(dict(spec=spec,criterion=float(np.mean(scores))))
        chosen=max(candidates,key=lambda x:x['criterion']);destination=folder/'selected.pt'
        shutil.copy2(folder/'predictive-best.pt',destination);chosen['spec']['checkpoint']=str(destination)
        selected[seed]=dict(**chosen,checkpoint_sha256=sha(destination),candidates=candidates)
        print('SELECT',stage_id,seed,chosen['criterion'],chosen['spec'],flush=True)
        with open('PROGRESS.md','a') as progress:progress.write(f"\n価値学習stage{stage_id} seed{seed}: {summary['steps']}更新/{summary['training_seconds']:.2f}秒、validation criterion {chosen['criterion']:+.3f}、alpha={chosen['spec']['alpha']}、gate={gate}。\n")
    write(root/'selection.json',dict(stage=stage_id,validation_sha256=prefixes['sha256'],selected=selected,training=training))
    return selected,training


def run(a):
    root=Path(a.output)
    if (root/'run-config.json').exists():raise ValueError('use fresh output')
    root.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    seeds=[201,202,203];validation=make_prefixes(91000,12)
    write(root/'validation-prefixes.json',validation)
    write(root/'run-config.json',dict(seeds=seeds,arguments=vars(a),source_hashes=source_hashes(),
          budget_ms=6,first_stage_steps=1600,second_stage_steps=2400,gpu_training_hard_upper_seconds=720,
          split=dict(train1=110000,train2=120000,validation=91000,holdout1=92000,holdout2=93000),
          revision='if initial stage does not clearly win across all seeds, collect separately continued search leaves and gate value to <=24 remaining pieces'))
    collect(root/'games.npz',games=a.data_games,seed=110000,workers=a.workers)
    first,train1=stage(root/'stage1',[root/'games.npz'],seeds,validation,1)
    holdout1=make_prefixes(92000,a.holdout_prefixes);write(root/'holdout1-prefixes.json',holdout1)
    specs={f'value-{s}':first[s]['spec'] for s in seeds};specs.update(static=dict(kind='static'),C=dict(kind='C'))
    pairs=[(f'value-{s}',b) for s in seeds for b in ('static','C')]+[('C','static')]
    report1=evaluate(specs,pairs,holdout1,root/'stage1/holdout-results.json')
    strong=all(r['summary']['difference_ci95'][0]>0 for r in report1['results'] if r['names'][0].startswith('value'))
    write(root/'stage1/diagnosis.json',dict(all_seed_baseline_score_ci_positive=strong,
          predictive_metrics=train1,decision='retain first stage' if strong else 'run branch-outcome data aggregation and later-game value gating'))
    train2=[]
    if not strong:
        print('REVISION: separately continued search leaves + late-game value gate',flush=True)
        with open('PROGRESS.md','a') as progress:progress.write('\nStage1で全seed・両対照へのCI正を達成せず。診断JSONを保存し、探索leafを独立に終局まで継続した追加データと後半局面gateでstage2へ進む。旧holdout92000は再選択に使わず、新holdout93000を使用。\n')
        collect(root/'branches.npz',games=max(600,a.data_games//2),seed=120000,workers=a.workers,branches=True)
        second,train2=stage(root/'stage2',[root/'games.npz',root/'branches.npz'],seeds,validation,2,
                            initial={s:first[s]['spec']['checkpoint'] for s in seeds},steps=2400,gate=24)
        # New holdout after revised checkpoints have been fixed; never re-use old holdout for selection.
        holdout2=make_prefixes(93000,a.holdout_prefixes);write(root/'holdout2-prefixes.json',holdout2)
        specs={f'value-{s}':second[s]['spec'] for s in seeds};specs.update(static=dict(kind='static'),C=dict(kind='C'))
        evaluate(specs,pairs,holdout2,root/'stage2/holdout-results.json')
    write(root/'run-summary.json',dict(wall_seconds=time.perf_counter()-begin,gpu_training_seconds=sum(t['invocation_seconds'] for t in train1+train2),
         stage1_training=train1,stage2_training=train2,completed=True))
    print('VALUE ROUND COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','collect','train','tournament','play'])
    p.add_argument('--output',default='artifacts/value-search/run');p.add_argument('--seed',type=int,default=201)
    p.add_argument('--data',nargs='+');p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--initialize')
    p.add_argument('--steps',type=int,default=1600);p.add_argument('--seconds',type=float,default=120)
    p.add_argument('--data-games',type=int,default=1800);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--branches',action='store_true');p.add_argument('--prefix-file');p.add_argument('--specs')
    p.add_argument('--holdout-prefixes',type=int,default=24);p.add_argument('--budget-ms',type=float,default=6)
    p.add_argument('--alpha',type=float,default=.5);p.add_argument('--gate',type=int,default=42);p.add_argument('--device',default='cuda')
    a=p.parse_args();seed_all(a.seed)
    if a.command=='run':run(a)
    elif a.command=='collect':collect(a.output,a.data_games,a.seed,a.workers,a.branches)
    elif a.command=='train':train(a.data,a.output,a.seed,a.steps,a.seconds,a.device,a.resume,a.initialize)
    elif a.command=='tournament':
        specs=json.loads(Path(a.specs).read_text());names=list(specs)
        evaluate(specs,[(names[i],names[j]) for i in range(len(names)) for j in range(i+1,len(names))],read_prefixes(a.prefix_file),a.output,a.budget_ms,a.device)
    else:
        agent=BudgetAgent('value',a.checkpoint,a.alpha,a.gate,a.budget_ms,a.device);s=State();rng=random.Random(a.seed)
        while not s.done:
            print(s.render())
            if s.turn==0:
                qs=legal_moves(s)
                if not qs:m=None
                else:
                    for i,q in enumerate(qs):print(i,q.move)
                    try:m=qs[int(input('index> '))].move
                    except (ValueError,IndexError):continue
            else:m=agent.act(s,rng)
            play(s,m,validate=m is not None)
        print('scores',s.score(0),s.score(1));print(agent.stats())
if __name__=='__main__':main()
