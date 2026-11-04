"""Grouped counterfactual terminal returns with shared continuation seeds."""
import concurrent.futures
import multiprocessing
import random
import time
from pathlib import Path
import numpy as np
from .engine import State,legal_moves,play
from .ai import seed_all,features,heuristic
from .research import Agent,clone,source_hashes,sha
from .value_search import encode,SCALE
from .value_experiment import write

AGENTS=None


def initialize_worker():
    global AGENTS
    seed_all(444)
    AGENTS=[Agent('heuristic:search','cpu'),Agent('artifacts/research-a2/selected.json:search','cpu')]


def continuation(leaf,agent,seed):
    s=clone(leaf);rng=random.Random(seed);perspective=s.turn;plies=0
    while not s.done:
        qs=legal_moves(s)
        move=rng.choice(qs).move if qs and rng.random()<.05 else agent.act(s,rng)
        play(s,move,validate=False);plies+=1
    return (s.score(perspective)-s.score(1-perspective))/SCALE,plies


def candidate_leaves(root,rng,count=4):
    qs=legal_moves(root)
    if len(qs)<count:return []
    x,_=features(root,qs);order=np.argsort(-heuristic(x),kind='stable')[:8]
    indices=rng.sample(list(order),count);leaves=[];moves=[]
    for i in indices:
        child=clone(root);move=qs[int(i)].move;play(child,move,validate=False)
        replies=legal_moves(child)
        reply=None
        if replies:
            rx,_=features(child,replies);reply=replies[int(np.argmax(heuristic(rx)))].move
        play(child,reply,validate=False)
        assert child.turn==root.turn
        leaves.append(child);moves.append([move.piece,move.orientation,move.row,move.col])
    return leaves,moves


def generate(task):
    seed,groups,repeats=task;rng=random.Random(seed);s=State();prefix=[]
    for _ in range(rng.randint(2,6)):
        m=rng.choice(legal_moves(s)).move;prefix.append([m.piece,m.orientation,m.row,m.col]);play(s,m)
    trace=[];agent=AGENTS[seed%2]
    while not s.done:
        qs=legal_moves(s)
        if sum(x.bit_count() for x in s.remaining)<=24 and len(qs)>=4:trace.append(clone(s))
        m=rng.choice(qs).move if qs and rng.random()<.08 else agent.act(s,rng)
        play(s,m,validate=False)
    rows=[];details=[];plies=0
    for group,root in enumerate(rng.sample(trace,min(groups,len(trace)))):
        leaves,moves=candidate_leaves(root,rng)
        targets=[];b=[];x=[];seeds=[seed*100+group*10+j for j in range(repeats)]
        for leaf in leaves:
            eb,ex=encode(leaf);b.append(np.packbits(eb.reshape(-1)));x.append(ex);returns=[]
            for j,roll_seed in enumerate(seeds):
                value,n=continuation(leaf,AGENTS[j%2],roll_seed);returns.append(value);plies+=n
            targets.append(returns)
        rows.append((b,x,targets));details.append(dict(game=seed,group=group,root=dict(boards=root.boards,remaining=root.remaining,last=root.last,turn=root.turn,passes=root.passes),moves=moves,continuation_seeds=seeds))
    return rows,details,prefix,plies


def collect(path,games=1200,seed=200000,workers=6,groups=2,repeats=2):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('dataset exists')
    start=time.perf_counter();boards=[];states=[];returns=[];ids=[];metadata=[];prefixes=[];plies=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,(rows,details,prefix,n) in enumerate(pool.map(generate,[(seed+i,groups,repeats) for i in range(games)],chunksize=4)):
            for b,x,y in rows:boards.append(b);states.append(x);returns.append(y);ids.append(seed+i)
            metadata.extend(details);prefixes.append(prefix);plies+=n
            if (i+1)%100==0:print('generated',i+1,'games',len(ids),'groups',round(time.perf_counter()-start,2),'seconds',flush=True)
    np.savez_compressed(path,board=np.array(boards),state=np.array(states),returns=np.array(returns,dtype=np.float32),game=np.array(ids))
    write(path.with_suffix('.groups.json'),metadata);write(path.with_suffix('.prefixes.json'),prefixes)
    summary=dict(games=games,groups=len(ids),candidates=len(ids)*4,continuations=len(ids)*4*repeats,continuation_plies=plies,seed=seed,workers=workers,repeats=repeats,seconds=time.perf_counter()-start,sha256=sha(path),source_hashes=source_hashes())
    write(path.with_suffix('.json'),summary);print(summary,flush=True);return summary
