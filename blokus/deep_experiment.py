"""Frozen weights; separated development, selection and fresh paired holdout."""
import argparse
import json
import random
import time
from pathlib import Path
from .ai import seed_all
from .engine import State,play,legal_moves
from .aligned_data import BASE
from .value_search import BudgetAgent
from .deep_search import DeepAgent
from .research import sha,source_hashes
from .tournament import make_prefixes,prefix_state,summarize,read_prefixes
from .visited_data import state_hash,state_dict
from .value_experiment import write

CONFIGS={
    'deep':dict(root_width=8,width=3,deep_gate=24,exact_pieces=8),
    'late':dict(root_width=12,width=3,deep_gate=14,exact_pieces=8),
}


def agent(spec,device='cuda'):
    if spec['kind']=='old':return BudgetAgent('value',BASE,.5,24,spec['budget_ms'],device)
    return DeepAgent(BASE,device=device,**{k:v for k,v in spec.items() if k!='kind'})


def match(left,right,fixture,device='cuda'):
    agents=[agent(left,device),agent(right,device)];games=[];certificates=[];start=time.perf_counter()
    for i,p in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            s=prefix_state(p);rngs=[random.Random(8877+i*2+j) for j in [0,1]];moves=[];hashes=[state_hash(s)];timings=[];details=[]
            while not s.done:
                who=0 if s.turn==seat else 1;a=agents[who];before=state_dict(s);m=a.act(s,rngs[s.turn]);timings.append([who,a.times[-1]*1000])
                detail=dict(a.decisions[-1]) if isinstance(a,DeepAgent) else None;details.append(detail)
                if detail and detail['exact']:
                    certificates.append(dict(prefix_id=p['id'],seat=seat,ply=len(moves),state=before,state_hash=state_hash(s),move=[m.piece,m.orientation,m.row,m.col] if m else None,value=detail['value'],who=who))
                # Legal placements checked independently; pass checked by enumeration.
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);moves.append([m.piece,m.orientation,m.row,m.col] if m else None);hashes.append(state_hash(s));assert len(moves)<=86
            games.append(dict(prefix_id=p['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decision_ms=timings,details=details))
    return dict(left=dict(spec=left,stats=agents[0].stats()),right=dict(spec=right,stats=agents[1].stats()),allowances_ms=[left['budget_ms'],right['budget_ms']],checkpoint=BASE,checkpoint_sha256=sha(BASE),prefix_sha256=fixture['sha256'],games=games,exact_certificates=certificates,summary=summarize(games),wall_seconds=time.perf_counter()-start)


def tournament(pairs,fixture,path):
    report=dict(source_hashes=source_hashes(),prefix_sha256=fixture['sha256'],results=[])
    for name,left,right in pairs:
        r=match(left,right,fixture);r['name']=name;report['results'].append(r);write(path,report)
        print('MATCH',name,r['summary'],'stats',r['left']['stats'],flush=True)
    return report


def run(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'config.json').exists():raise ValueError('existing experiment')
    seed_all(701);begin=time.perf_counter()
    write(root/'config.json',dict(configs=CONFIGS,checkpoint=BASE,checkpoint_sha256=sha(BASE),budgets=[25,100],dev_seed=510000,validation_seed=520000,holdout_seed=530000,dev_prefixes=32,validation_prefixes=32,holdout_prefixes=96,training=False,source_hashes=source_hashes()))
    dev=make_prefixes(510000,32);write(root/'dev-prefixes.json',dev)
    pairs=[(f'{name}-{ms}',dict(kind='deep',budget_ms=ms,**cfg),dict(kind='old',budget_ms=ms)) for name,cfg in CONFIGS.items() for ms in [25,100]]
    development=tournament(pairs,dev,root/'dev-results.json')
    best_name=max(CONFIGS,key=lambda name:sum(r['summary']['mean_difference'] for r in development['results'] if r['name'].startswith(name+'-'))/2)
    write(root/'dev-selection.json',dict(config=best_name,criterion='mean same-budget point difference over25/100ms',frozen_config=CONFIGS[best_name]))
    with open('PROGRESS.md','a') as f:f.write(f'\nDeep development selected config={best_name}; next newvalidation520000.\n')
    small=make_prefixes(510999,8);write(root/'six-prefixes.json',small)
    tournament([(best_name+'-6',dict(kind='deep',budget_ms=6,**CONFIGS[best_name]),dict(kind='old',budget_ms=6))],small,root/'six-results.json')
    validation=make_prefixes(520000,32);write(root/'validation-prefixes.json',validation)
    vr=tournament([(f'{best_name}-{ms}',dict(kind='deep',budget_ms=ms,**CONFIGS[best_name]),dict(kind='old',budget_ms=ms)) for ms in [25,100]],validation,root/'validation-results.json')
    choice=max(vr['results'],key=lambda r:r['summary']['mean_difference']);spec=choice['left']['spec'];write(root/'selection.json',dict(spec=spec,criterion='newvalidation same-budget mean point difference',validation_mean=choice['summary']['mean_difference'],validation_prefix_sha256=validation['sha256']))
    with open('PROGRESS.md','a') as f:f.write(f'\nDeep final selection frozen: {spec}, validation mean={choice["summary"]["mean_difference"]}; fresh530000 begins.\n')
    hold=make_prefixes(530000,96);write(root/'holdout-prefixes.json',hold)
    tournament([('selected-vs-old',spec,dict(kind='old',budget_ms=spec['budget_ms']))],hold,root/'holdout-results.json')
    write(root/'summary.json',dict(completed=True,seconds=time.perf_counter()-begin,selected=spec,checkpoint_sha256=sha(BASE),training_updates=0));print('COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','tournament','play']);p.add_argument('--output',default='artifacts/deep-search');p.add_argument('--spec');p.add_argument('--prefix-file');p.add_argument('--device',default='cuda');a=p.parse_args()
    if a.command=='run':run(a.output)
    elif a.command=='tournament':
        spec=json.loads(Path(a.spec).read_text())['spec'];tournament([('selected-vs-old',spec,dict(kind='old',budget_ms=spec['budget_ms']))],read_prefixes(a.prefix_file),a.output)
    else:
        spec=json.loads(Path(a.spec).read_text())['spec'];bot=agent(spec,a.device);s=State();rng=random.Random(0)
        while not s.done:
            print(s.render())
            if s.turn==0:
                qs=legal_moves(s);m=None
                if qs:
                    for i,q in enumerate(qs):print(i,q.move)
                    try:m=qs[int(input('index> '))].move
                    except (ValueError,IndexError):continue
            else:m=bot.act(s,rng)
            play(s,m,validate=m is not None)
        print('scores',s.score(0),s.score(1));print(bot.stats())
if __name__=='__main__':main()
