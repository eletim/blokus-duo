"""A2: CPU cross-entropy-method policy optimization by actual paired games."""
import argparse
import itertools
import json
import shutil
import time
from pathlib import Path
import numpy as np
from .ai import seed_all
from .research import sha, source_hashes
from .tournament import make_prefixes, read_prefixes, match, tournament

INITIAL = np.array([12,4,2,1.2,2,.3,1,.2,-.6,0,0,.2],dtype=np.float64)
# Fix positive scale (piece-size weight=12). State-only features cannot rank moves.
ACTIVE = np.array([1,2,3,4,5,6,7,8,11])


def write(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2))


def linear_checkpoint(path,weights,metadata):
    write(path,dict(format=3,method='A2',weights=np.asarray(weights).tolist(),metadata=metadata))


def population(mean,std,incumbent,rng,count):
    if count<4: raise ValueError('population >=4 required')
    proposals=[incumbent.copy(),mean.copy()]
    while len(proposals)<count:
        z=rng.normal(size=len(ACTIVE))
        for sign in (1,-1):
            if len(proposals)>=count:break
            w=mean.copy();w[ACTIVE]=np.clip(mean[ACTIVE]+sign*std*z,-20,20);proposals.append(w)
    return proposals


def update_distribution(proposals,fitness,mean,std,elite=3):
    # Stable ordering breaks fitness ties independently of random hash order.
    idx=np.argsort(-np.asarray(fitness),kind='stable')[:elite]
    top=np.asarray([proposals[i] for i in idx])
    new_mean=mean.copy();new_mean[ACTIVE]=.4*mean[ACTIVE]+.6*top[:,ACTIVE].mean(0)
    new_std=np.maximum(.25,.4*std+.6*top[:,ACTIVE].std(0))
    return new_mean,new_std,proposals[int(idx[0])].copy()


def train(output,baseline,seconds=240,generations=100,count=10,prefix_count=3,seed=991,resume=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'cem-state.json').exists() and not resume:raise ValueError('existing CEM run; use --resume')
    if seconds<=0 or generations<1 or prefix_count<1:raise ValueError('positive budget required')
    seed_all(seed);rng=np.random.default_rng(seed);mean=INITIAL.copy();std=np.full(len(ACTIVE),2.0);incumbent=mean.copy()
    generation=0;games=0;previous_seconds=0.;last_seconds=0.;evaluations=0
    if resume:
        state=json.loads(Path(resume).read_text());mean=np.array(state['mean']);std=np.array(state['std']);incumbent=np.array(state['incumbent'])
        rng.bit_generator.state=state['rng'];generation=state['generation'];games=state['games'];evaluations=state['evaluations'];previous_seconds=state['training_seconds']
    config=dict(algorithm='CPU CEM, paired score difference directly; no teacher/GPU gradients',seconds=seconds,max_generations=generations,population=count,
                train_prefix_count=prefix_count,prefix_seed_base=71000,seed=seed,baseline=str(baseline),baseline_sha256=sha(baseline),active=ACTIVE.tolist(),
                fixed_piece_size_weight=12,opponents=['heuristic','v1'],elite=3,mean_smoothing=.6,std_floor=.25,device='cpu',source_hashes=source_hashes(),
                budget='check 1.25*previous generation time before starting; no generation interrupted; first generation always completes')
    write(root/'config.json',config)
    if not resume:linear_checkpoint(root/'untrained.json',INITIAL,config)
    begin=time.perf_counter();initial_generation=generation;stored=[]
    with (root/'train.jsonl').open('a') as log:
        while generation-initial_generation<generations:
            elapsed=time.perf_counter()-begin
            if elapsed>=seconds or (last_seconds and elapsed+1.25*last_seconds>seconds):break
            t=time.perf_counter();fixture=make_prefixes(71000+generation,prefix_count)
            write(root/'train-prefixes'/f'{generation+1:04}.json',fixture)
            proposals=population(mean,std,incumbent,rng,count);fitness=[];details=[]
            for i,w in enumerate(proposals):
                path=root/'population'/f'{i:02}.json';linear_checkpoint(path,w,dict(generation=generation+1,candidate=i))
                per_opponent=[]
                for opponent in ('heuristic',str(baseline)):
                    r=match(str(path),opponent,fixture,device='cpu')
                    per_opponent.append(dict(opponent=opponent,summary=r['summary'],games=r['games']))
                    games+=len(r['games'])
                fit=float(np.mean([r['summary']['mean_difference'] for r in per_opponent]))
                fitness.append(fit);evaluations+=1;details.append(dict(weights=w.tolist(),fitness=fit,results=per_opponent))
            mean,std,incumbent=update_distribution(proposals,fitness,mean,std);generation+=1
            last_seconds=time.perf_counter()-t
            record=dict(generation=generation,seconds=last_seconds,training_seconds=previous_seconds+time.perf_counter()-begin,games=games,
                        evaluations=evaluations,prefix_sha256=fixture['sha256'],mean=mean.tolist(),std=std.tolist(),incumbent=incumbent.tolist(),candidates=details)
            log.write(json.dumps(record)+'\n');log.flush()
            state=dict(generation=generation,mean=mean.tolist(),std=std.tolist(),incumbent=incumbent.tolist(),rng=rng.bit_generator.state,
                       training_seconds=previous_seconds+time.perf_counter()-begin,games=games,evaluations=evaluations)
            write(root/'cem-state.json',state)
            linear_checkpoint(root/'latest.json',mean,dict(config=config,**state))
            if generation%10==0:
                target=root/f'generation-{generation:04}.json';shutil.copy2(root/'latest.json',target);stored.append(str(target))
            print(json.dumps({k:record[k] for k in ['generation','seconds','training_seconds','games','evaluations']}|dict(best_fitness=max(fitness))),flush=True)
        if generation==initial_generation:raise RuntimeError('no complete generation')
    summary=dict(training_seconds=previous_seconds+time.perf_counter()-begin,invocation_seconds=time.perf_counter()-begin,generations=generation,games=games,
                 candidate_evaluations=evaluations,checkpoints=stored,device='cpu',gpu_gradient_updates=0,config=config)
    write(root/'summary.json',summary);return summary


def run(a):
    root=Path(a.output)
    if root.exists() and any(root.iterdir()):raise ValueError('new empty output directory required')
    root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    previous=Path(a.round1);selection=json.loads((previous/'selection.json').read_text())
    # Select the previous method ONLY by original validation, never old holdout.
    best=max(selection['selected'],key=lambda m:selection['selected'][m]['criterion'])
    validation=read_prefixes(previous/'validation-prefixes.json');holdout=make_prefixes(61000,24)
    old_holdout=read_prefixes(previous/'holdout-prefixes.json')
    assert not {p['id'] for p in holdout['prefixes']} & {p['id'] for p in validation['prefixes']+old_holdout['prefixes']}
    write(root/'holdout-prefixes.json',holdout);write(root/'validation-prefixes.json',validation)
    write(root/'run-config.json',dict(arguments=vars(a),baseline_method_from_original_validation=best,source_hashes=source_hashes(),new_holdout_sha256=holdout['sha256']))
    summary=train(root/'train',previous/'v1.pt',a.seconds,a.generations,a.population,a.train_prefixes,a.seed)
    choices=sorted((root/'train').glob('generation-*.json'))+[root/'train/latest.json']
    records=[]
    for path in choices:
        results=[match(str(path),opp,validation,device='cpu') for opp in ('heuristic',str(previous/'v1.pt'))]
        records.append(dict(checkpoint=str(path),criterion=float(np.mean([r['summary']['mean_difference'] for r in results])),results=results))
        write(root/'validation-results.json',records)
    winner=max(records,key=lambda r:r['criterion']);shutil.copy2(winner['checkpoint'],root/'selected.json')
    write(root/'selection.json',dict(checkpoint=str(root/'selected.json'),sha256=sha(root/'selected.json'),source=winner['checkpoint'],criterion=winner['criterion'],
                                   validation_sha256=validation['sha256'],holdout_sha256=holdout['sha256'],baseline_method=best,selection_before_holdout=True))
    specs=dict(A2=str(root/'selected.json'),heuristic='heuristic',v1=str(previous/'v1.pt'),**{best:selection['selected'][best]['checkpoint']},
               C_search=selection['selected']['C']['checkpoint']+':search')
    pairs=list(itertools.combinations(specs,2))
    specs['A2_untrained']=str(root/'train/untrained.json');pairs.append(('A2','A2_untrained'))
    result=tournament(specs,pairs,holdout,root/'holdout-results.json',device=a.device)
    write(root/'run-summary.json',dict(wall_seconds=time.perf_counter()-start,training=summary,matches=len(result['results']),completed=True,
                                     selection_sha256=sha(root/'selection.json'),holdout_sha256=sha(root/'holdout-results.json')))
    print('A2 COMPLETE',root,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','train'])
    p.add_argument('--output',default='artifacts/research-a2');p.add_argument('--round1',default='artifacts/research-round1')
    p.add_argument('--baseline',default='artifacts/research-round1/v1.pt');p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--generations',type=int,default=100);p.add_argument('--population',type=int,default=10)
    p.add_argument('--train-prefixes',type=int,default=3);p.add_argument('--seed',type=int,default=991)
    p.add_argument('--device',default='cuda');p.add_argument('--resume')
    a=p.parse_args();seed_all(a.seed)
    if a.command=='run':run(a)
    else:train(a.output,a.baseline,a.seconds,a.generations,a.population,a.train_prefixes,a.seed,a.resume)
if __name__=='__main__':main()
