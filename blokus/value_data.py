"""Parallel CPU terminal-outcome data; branches are separately played to completion."""
import concurrent.futures
import multiprocessing
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from .engine import State,legal_moves,play
from .ai import seed_all,features,heuristic
from .research import Agent,clone,source_hashes
from .value_search import encode,SCALE

_WORKER_AGENTS=None


def initialize_worker():
    global _WORKER_AGENTS
    seed_all(818)
    _WORKER_AGENTS=[Agent('heuristic','cpu'),Agent('heuristic:search','cpu'),
                    Agent('artifacts/research-a2/selected.json:search','cpu')]


def finish(s,agent,rng):
    while not s.done:play(s,agent.act(s,rng),validate=False)
    return [s.score(0),s.score(1)]


def generate_game(task):
    seed,branches=task;rng=random.Random(seed);s=State();agent=_WORKER_AGENTS[seed%len(_WORKER_AGENTS)]
    prefix=[]
    for _ in range(rng.randint(2,6)):
        qs=legal_moves(s);m=rng.choice(qs).move
        prefix.append([m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
    trace=[]
    while not s.done:
        trace.append(clone(s))
        qs=legal_moves(s)
        if qs and rng.random()<.12:m=rng.choice(qs).move
        else:m=agent.act(s,rng)
        play(s,m,validate=False)
    scores=[s.score(0),s.score(1)]
    rows=[(st,(scores[st.turn]-scores[1-st.turn])/SCALE,0) for st in trace]
    # Query off-policy leaf states; their labels come from their OWN continuations.
    if branches:
        usable=[st for st in trace if legal_moves(st)]
        for _ in range(4):
            st=clone(rng.choice(usable));qs=legal_moves(st);x,_=features(st,qs)
            order=np.argsort(-heuristic(x),kind='stable')[:8]
            play(st,qs[int(rng.choice(order))].move,validate=False)
            replies=legal_moves(st)
            if replies:
                x,_=features(st,replies);order=np.argsort(-heuristic(x),kind='stable')[:3]
                play(st,replies[int(rng.choice(order))].move,validate=False)
            else:play(st,None,validate=False)
            leaf=clone(st);result=finish(st,_WORKER_AGENTS[2],rng)
            # Duplicate branches in the sampler later, never fake labels for sibling states.
            rows.append((leaf,(result[leaf.turn]-result[1-leaf.turn])/SCALE,1))
    boards=[];states=[];targets=[];types=[]
    for st,y,kind in rows:
        b,x=encode(st);boards.append(np.packbits(b.reshape(-1)));states.append(x);targets.append(y);types.append(kind)
    return np.array(boards),np.array(states),np.array(targets,dtype=np.float32),np.array(types,dtype=np.uint8),seed,prefix


def collect(path,games=1800,seed=110000,workers=6,branches=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('dataset already exists')
    start=time.perf_counter();bs=[];xs=[];ys=[];ks=[];gs=[];prefixes=[]
    # Map preserves seed order even though worker completion order differs.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,(b,x,y,k,g,prefix) in enumerate(pool.map(generate_game,[(seed+i,branches) for i in range(games)],chunksize=8)):
            bs.append(b);xs.append(x);ys.append(y);ks.append(k);gs.append(np.full(len(y),g,dtype=np.int64));prefixes.append(prefix)
            if (i+1)%200==0:print('generated',i+1,'games',round(time.perf_counter()-start,2),'seconds',flush=True)
    np.savez_compressed(path,board=np.concatenate(bs),state=np.concatenate(xs),target=np.concatenate(ys),kind=np.concatenate(ks),game=np.concatenate(gs))
    prefix_path=path.with_suffix('.prefixes.json');prefix_path.write_text(json.dumps(prefixes))
    meta=dict(games=games,positions=sum(map(len,ys)),seed=seed,workers=workers,branches=branches,seconds=time.perf_counter()-start,
              sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prefix_sha256=hashlib.sha256(prefix_path.read_bytes()).hexdigest(),source_hashes=source_hashes())
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,default=str),flush=True)
    return meta
