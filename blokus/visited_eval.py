"""Paired evaluation with per-agent budgets and replayable middle/endgame hashes."""
import random
import time
from .engine import play
from .value_search import BudgetAgent
from .research import sha,source_hashes
from .tournament import prefix_state,summarize
from .value_experiment import write
from .visited_data import state_hash


def match(left,right,fixture,device='cuda'):
    specs=[dict(left),dict(right)];budgets=[s.pop('budget_ms',6) for s in specs]
    agents=[BudgetAgent(**s,budget_ms=b,device=device) for s,b in zip(specs,budgets)];games=[];begin=time.perf_counter()
    for i,prefix in enumerate(fixture['prefixes']):
        for seat in [0,1]:
            s=prefix_state(prefix);rngs=[random.Random(712+i*2+p) for p in [0,1]];moves=[];hashes=[state_hash(s)];timings=[]
            while not s.done:
                who=0 if s.turn==seat else 1;m=agents[who].act(s,rngs[s.turn]);timings.append([who,agents[who].times[-1]*1000])
                play(s,m);moves.append([m.piece,m.orientation,m.row,m.col] if m else None);hashes.append(state_hash(s));assert len(moves)<=86
            games.append(dict(prefix_id=prefix['id'],seat=seat,scores=[s.score(0),s.score(1)],difference=s.score(seat)-s.score(1-seat),moves=moves,state_hashes=hashes,decision_ms=timings))
    def meta(spec,agent):
        d=dict(spec=spec,stats=agent.stats())
        if spec.get('checkpoint'):d['checkpoint_sha256']=sha(spec['checkpoint'])
        return d
    return dict(left=meta(left,agents[0]),right=meta(right,agents[1]),allowances_ms=budgets,prefix_sha256=fixture['sha256'],games=games,summary=summarize(games),wall_seconds=time.perf_counter()-begin)


def tournament(specs,pairs,fixture,path):
    report=dict(prefix_sha256=fixture['sha256'],source_hashes=source_hashes(),results=[])
    for a,b in pairs:
        r=match(specs[a],specs[b],fixture);r['names']=[a,b];report['results'].append(r);write(path,report)
        print('MATCH',a,b,r['summary'],flush=True)
    return report
