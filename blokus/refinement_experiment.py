"""Finite pilot, cold paired validation, and gated fresh holdout."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from .refinement_search import RefinementAgent,LimitedTree
from .value_search import BudgetAgent,ValueEvaluator
from .reply_search import reset_game
from .aligned_data import BASE
from .engine import State,play,legal_moves
from .research import clone,sha,source_hashes
from .visited_data import state_hash,state_dict
from .node_diagnostic import row,child
from .tournament import make_prefixes,prefix_state,summarize,read_prefixes
from .value_experiment import write
from .ai import seed_all
ROOT=Path('artifacts/minimax-refinement')

def agent(spec):
    if spec['method']=='old':return BudgetAgent('value',BASE,.5,24,spec['budget_ms'],'cuda')
    return RefinementAgent(BASE,spec['budget_ms'],spec['method'],reserve_fraction=spec.get('reserve_fraction',0))

def match(left,right,fixture):
    agents=[agent(left),agent(right)];games=[];begin=time.perf_counter()
    for i,p in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            for a in agents:
                if isinstance(a,RefinementAgent):a.reset_game()
                else:reset_game(a)
            s=prefix_state(p);rngs=[random.Random(830000+i*2+j) for j in [0,1]];moves=[];hashes=[state_hash(s)];decisions=[]
            while not s.done:
                who=0 if s.turn==seat else 1;a=agents[who];n=a.roots;m=a.act(s,rngs[s.turn])
                z=dict(a.decisions[-1]) if isinstance(a,RefinementAgent) else dict(ms=a.times[-1]*1000,completed_roots=a.roots-n)
                z['who']=who;z['remaining_pieces']=sum(v.bit_count() for v in s.remaining);decisions.append(z)
                if m is None:assert not legal_moves(s)
                play(s,m,validate=m is not None);moves.append(row(m));hashes.append(state_hash(s));assert len(moves)<=86
            games.append(dict(prefix_id=p['id'],seat=seat,cold=True,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decisions=decisions))
    return dict(left=dict(spec=left,stats=agents[0].stats()),right=dict(spec=right,stats=agents[1].stats()),games=games,prefix_sha256=fixture['sha256'],checkpoint_sha256=sha(BASE),summary=summarize(games),seconds=time.perf_counter()-begin)

def states():
    rows=[]
    for i in range(12):
        s=State();rng=random.Random(820000+i);history=[];moves=[]
        while not s.done:
            history.append(clone(s));qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=False);moves.append(row(m))
        for kind,index in [('mid',len(history)//2),('end',len(history)-4)]:
            q=history[index];rows.append(dict(game=i,kind=kind,state=state_dict(q),state_hash=state_hash(q),game_moves=moves,ply=index))
    write(ROOT/'states.json',rows);return rows

def pilot(revision=0):
    from tests.test_deep_search import oracle
    start=time.perf_counter();ss=states() if revision==0 else json.loads((ROOT/'states.json').read_text());out=[]
    reserve=0 if revision==0 else .35
    for r in ss:
        s=clone(State(**r['state']));z={k:v for k,v in r.items() if k not in ('state','game_moves')};z['fixed']=[];z['timed']=[]
        for method in ['control','uniform','priority']:
            a=RefinementAgent(BASE,100,method,node_limit=2048,reserve_fraction=reserve)
            m=a.act(s,random.Random(91));first=(m,dict(a.last_values),a.decisions[-1]['nodes']);a.reset_game();m2=a.act(s,random.Random(91));assert first==(m2,a.last_values,a.decisions[-1]['nodes'])
            z['fixed'].append(dict(method=method,move=row(m),values=a.last_values,**a.decisions[-1]))
        if r['kind']=='end':
            try:
                v,best,n=oracle(s);z['oracle']=dict(value=v,best=[row(m) for m in best],nodes=n)
                t=LimitedTree(ValueEvaluator(None,'cpu'));tv,tm=t.root(s,[q.move for q in legal_moves(s)] or [None],0,True);assert tv==v and tm in best
                for f in z['fixed']:
                    from .engine import Move
                    c=child(s,Move(*f['move']) if f['move'] else None)
                    cv=(c.score(c.turn)-c.score(1-c.turn)) if c.done else oracle(c)[0]
                    f['exact_regret']=v+cv
            except RuntimeError:z['oracle']=dict(capped=True)
        for budget in [25,100]:
            for method in ['old','control','uniform','priority']:
                a=agent(dict(method=method,budget_ms=budget,reserve_fraction=reserve));m=a.act(s,random.Random(91))
                d=dict(a.decisions[-1]) if isinstance(a,RefinementAgent) else dict(ms=a.times[-1]*1000,completed_roots=a.roots)
                z['timed'].append(dict(method=method,budget_ms=budget,move=row(m),**d))
        out.append(z);write(ROOT/f'pilot-{revision}.json',out)
    summary=[]
    for b in [25,100]:
        ds=[t for r in out if r['kind']=='mid' for t in r['timed'] if t['budget_ms']==b and t['method']=='priority']
        summary.append(dict(budget_ms=b,extra_fraction=float(np.mean([d['refined']>0 for d in ds])),initial_complete_fraction=float(np.mean([d['initial_complete'] for d in ds])),overrun_fraction=float(np.mean([d['ms']>b for d in ds])),mean_ms=float(np.mean([d['ms'] for d in ds])),mean_initial_ms=float(np.mean([d['sections_ms']['initial'] for d in ds])),mean_refined=float(np.mean([d['refined'] for d in ds]))))
    result=dict(revision=revision,reserve_fraction=reserve,summary=summary,seconds=time.perf_counter()-start);write(ROOT/f'pilot-summary-{revision}.json',result);print(result,flush=True)
    return result

def validation(revision,budget):
    reserve=0 if revision==0 else .35;fixture=make_prefixes(821000+100*revision,16);write(ROOT/f'validation-prefixes-{revision}.json',fixture)
    spec=dict(method='priority',budget_ms=budget,reserve_fraction=reserve);old=dict(method='old',budget_ms=budget);results=[]
    for right in [old,dict(method='uniform',budget_ms=budget,reserve_fraction=reserve),dict(method='control',budget_ms=budget,reserve_fraction=reserve)]:
        r=match(spec,right,fixture);results.append(r);write(ROOT/f'validation-results-{revision}.json',results);print('validation',revision,right,r['summary'],flush=True)
    return results

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['pilot','validation','holdout','play']);p.add_argument('--revision',type=int,default=0);p.add_argument('--budget',type=int,default=100);a=p.parse_args();seed_all(901)
    if a.command=='pilot':return pilot(a.revision)
    if a.command=='validation':return validation(a.revision,a.budget)
    if a.command=='holdout':
        spec=json.loads((ROOT/'selection.json').read_text())['spec'];fixture=make_prefixes(822000,96);write(ROOT/'holdout-prefixes.json',fixture);results=[]
        for right in [dict(method='old',budget_ms=spec['budget_ms']),dict(**{**spec,'method':'control'})]:
            r=match(spec,right,fixture);results.append(r);write(ROOT/'holdout-results.json',results);print('holdout',right,r['summary'],flush=True)
        return
    s=State();bot=agent(dict(method='priority',budget_ms=a.budget,reserve_fraction=0 if a.revision==0 else .35));rng=random.Random(0)
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
    print('scores',s.score(0),s.score(1))
if __name__=='__main__':main()
