"""Separated development choice, cold validation, fresh paired holdout."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from .rollout_search import RolloutAgent
from .value_search import BudgetAgent
from .reply_search import reset_game
from .aligned_data import BASE
from .engine import State,play,legal_moves
from .visited_data import state_hash
from .research import sha,source_hashes
from .tournament import make_prefixes,prefix_state,summarize,read_prefixes
from .value_experiment import write
from .ai import seed_all

def agent(spec):
    if spec['method']=='old':return BudgetAgent('value',BASE,.5,24,spec['budget_ms'],'cuda')
    return RolloutAgent(spec['budget_ms'],spec['method'],spec['diverse'])

def match(left,right,fixture):
    agents=[agent(left),agent(right)];games=[];begin=time.perf_counter()
    for i,p in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            for j,a in enumerate(agents):
                colour=seat if j==0 else 1-seat
                if isinstance(a,RolloutAgent):a.reset_game(760000+i*2+colour)
                else:reset_game(a)
            cold=[dict(turn=a.turn,seed=a.seed) if isinstance(a,RolloutAgent) else dict(cost_estimate=a.cost_estimate) for a in agents]
            s=prefix_state(p);rngs=[random.Random(770000+i*2+j) for j in [0,1]];moves=[];hashes=[state_hash(s)];decisions=[]
            while not s.done:
                who=0 if s.turn==seat else 1;a=agents[who];n=getattr(a,'roots',0);m=a.act(s,rngs[s.turn])
                z=dict(a.decisions[-1]) if isinstance(a,RolloutAgent) else dict(ms=a.times[-1]*1000,completed_roots=a.roots-n)
                z['who']=who;z['remaining_pieces']=sum(v.bit_count() for v in s.remaining);decisions.append(z)
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);moves.append([m.piece,m.orientation,m.row,m.col] if m else None);hashes.append(state_hash(s));assert len(moves)<=86
            games.append(dict(prefix_id=p['id'],seat=seat,cold=cold,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decisions=decisions))
    return dict(left=dict(spec=left,stats=agents[0].stats()),right=dict(spec=right,stats=agents[1].stats()),games=games,prefix_sha256=fixture['sha256'],checkpoint_sha256=sha(BASE),cache='cold every game; no TT; old scheduling estimates reset',summary=summarize(games),seconds=time.perf_counter()-begin)

def run(root):
    root=Path(root);seed_all(902);start=time.perf_counter()
    if (root/'validation-results.json').exists():raise ValueError('existing validation')
    pilot=json.loads((root/'pilot.json').read_text());assert not pilot['stop']
    ds=json.loads((root/'diagnostic.json').read_text());criterion={}
    for diverse in [False,True]:
        losses=[]
        for d in ds:
            if d['kind']!='mid':continue
            best=max(r['mean'] for r in d['references']);by={tuple(r['move']) if r['move'] else None:r['mean'] for r in d['references']}
            losses.append(np.mean([best-by[tuple(z['move']) if z['move'] else None] for z in d['experiments'] if z['diverse']==diverse]))
        criterion[str(diverse)]=float(np.mean(losses))
    diverse=min([False,True],key=lambda x:criterion[str(x)]);budget=100
    write(root/'dev-selection.json',dict(diverse=diverse,budget_ms=budget,criterion='minimum union-reference regret averaged over methods/seeds then source games (mid)',values=criterion,pilot='25ms coverage1/12;100ms coverage12/12 and median>=16'))
    validation=make_prefixes(721000,32);write(root/'validation-prefixes.json',validation);results=[];old=dict(method='old',budget_ms=budget)
    for method in ['uniform','halving']:
        spec=dict(method=method,budget_ms=budget,diverse=diverse);r=match(spec,old,validation);results.append(r);write(root/'validation-results.json',results);print('VALIDATION',method,r['summary'],flush=True)
    best=max(results,key=lambda r:r['summary']['mean_difference']);spec=best['left']['spec'];write(root/'selection.json',dict(spec=spec,criterion='newvalidation mean point difference',prefix_sha256=validation['sha256']))
    with open('PROGRESS.md','a') as f:f.write(f'\nRollout final selection {spec}; fresh722000/96 begins.\n')
    hold=make_prefixes(722000,96);write(root/'holdout-prefixes.json',hold);results=[]
    for name,left,right in [('selected-vs-old',spec,old),('halving-vs-uniform',dict(method='halving',budget_ms=budget,diverse=diverse),dict(method='uniform',budget_ms=budget,diverse=diverse))]:
        r=match(left,right,hold);r['name']=name;results.append(r);write(root/'holdout-results.json',results);print('HOLDOUT',name,r['summary'],flush=True)
    write(root/'run-summary.json',dict(seconds=time.perf_counter()-start,games=512,source_hashes=source_hashes(),training_updates=0))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','play','tournament']);p.add_argument('--root',default='artifacts/terminal-rollouts');p.add_argument('--prefix-file');p.add_argument('--output',default='artifacts/rollout-replay.json');a=p.parse_args()
    if a.command=='run':return run(a.root)
    spec=json.loads((Path(a.root)/'selection.json').read_text())['spec']
    if a.command=='tournament':return write(a.output,match(spec,dict(method='old',budget_ms=spec['budget_ms']),read_prefixes(a.prefix_file)))
    bot=agent(spec);bot.reset_game(10);s=State();rng=random.Random(0)
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
