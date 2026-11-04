"""Capture completed 6ms search evaluations, then deterministically label them."""
import concurrent.futures
import hashlib
import json
import multiprocessing
import random
import time
from pathlib import Path
import numpy as np
from .ai import seed_all
from .engine import State,legal_moves,play
from .research import sha,source_hashes
from .value_search import BudgetAgent,encode,exact_terminal,score_value
from .value_experiment import write
from .aligned_data import BASE,initialize,finish


def state_dict(s):return dict(boards=list(s.boards),remaining=list(s.remaining),last=list(s.last),turn=s.turn,passes=s.passes)


def state_hash(s):
    # Ignore irrelevant last-piece identities while retaining actual bonus scores.
    key=[s.boards,s.remaining,[s.score(0),s.score(1)],s.turn,s.passes]
    return hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()


def gather(path,roots=512,seed=420000):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('visited fixture exists')
    seed_all(seed);begin=time.perf_counter();agent=BudgetAgent('value',BASE,.5,24,6,'cuda');original=agent.evaluator.batch;captured=[]
    def capture(states):captured.append(tuple(states));return original(states)
    agent.evaluator.batch=capture;records=[];games=[];seen=set();g=0
    while len(records)<roots:
        if g>=roots*2 or time.perf_counter()-begin>1200:
            write(path.with_suffix('.partial.json'),dict(records=records,games=games,reason='collection wall/game bound'))
            raise RuntimeError('bounded collection could not obtain requested roots')
        rng=random.Random(seed+g);s=State();moves=[];trajectory=[state_hash(s)]
        for _ in range(rng.randint(2,6)):
            m=rng.choice(legal_moves(s)).move;play(s,m);moves.append([m.piece,m.orientation,m.row,m.col]);trajectory.append(state_hash(s))
        prefix=list(moves);eligible=[]
        while not s.done:
            captured.clear();before=agent.roots;move=agent.act(s,rng);completed=agent.roots-before
            # The evaluator may have seen one late child: only retain completed ones.
            if completed>=4 and sum(x.bit_count() for x in s.remaining)<=24:
                h=state_hash(s)
                if h not in seen:
                    leaves=[[state_dict(x) for x in branch] for branch in captured[:4]]
                    eligible.append(dict(game=seed+g,root=state_dict(s),root_hash=h,leaves=leaves,
                        leaf_hashes=[[state_hash(x) for x in branch] for branch in captured[:4]],completed_candidates=completed))
            qs=legal_moves(s)
            if qs and rng.random()<.08:move=rng.choice(qs).move
            play(s,move);moves.append([move.piece,move.orientation,move.row,move.col] if move else None);trajectory.append(state_hash(s))
        for row in rng.sample(eligible,min(12,len(eligible))):
            if len(records)>=roots:break
            if row['root_hash'] not in seen:records.append(row);seen.add(row['root_hash'])
        games.append(dict(game=seed+g,prefix=prefix,moves=moves,state_hashes=trajectory));g+=1
        if g%20==0:print('visited',g,'games',len(records),'roots',round(time.perf_counter()-begin,2),flush=True)
    result=dict(records=records,games=games,seed=seed,seconds=time.perf_counter()-begin,agent_stats=agent.stats(),source_hashes=source_hashes(),baseline_sha256=sha(BASE))
    write(path,result);return result


def stratified_subset(rows,n):
    # Spread samples across remaining-piece count and game id, deterministic.
    order=sorted(range(len(rows)),key=lambda i:(sum(x.bit_count() for x in rows[i]['root']['remaining']),rows[i]['game'],rows[i]['root_hash']))
    return [rows[order[int(i)]] for i in np.linspace(0,len(order)-1,n,dtype=int)]


def label_one(task):
    row,repeats=task;b=np.zeros((4,3,147),np.uint8);x=np.zeros((4,3,49),np.float32)
    y=np.zeros((4,3,repeats),np.float32);mask=np.zeros((4,3),bool);term=np.zeros((4,3),bool);scores=np.zeros((4,3),np.float32);plies=0;count=0
    base_seed=int(row['root_hash'][:15],16)
    for k,branch in enumerate(row['leaves']):
        for j,data in enumerate(branch):
            s=State(**data);eb,ex=encode(s);b[k,j]=np.packbits(eb.reshape(-1));x[k,j]=ex;mask[k,j]=True;term[k,j]=exact_terminal(s);scores[k,j]=score_value(s)
            for r in range(repeats):y[k,j,r],n=finish(s,base_seed+r,r);plies+=n;count+=1
    return b,x,y,mask,term,scores,row['game'],plies,count


def label(path,rows,repeats=4,workers=6):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('dataset exists')
    start=time.perf_counter();columns=[[] for _ in range(7)];plies=0;count=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(label_one,[(row,repeats) for row in rows],chunksize=2)):
            for col,value in zip(columns,result[:7]):col.append(value)
            plies+=result[7];count+=result[8]
            if (i+1)%100==0:print('labels',path.name,i+1,round(time.perf_counter()-start,2),flush=True)
    np.savez_compressed(path,**dict(zip(['board','state','returns','mask','terminal','score','game'],columns)))
    write(path.with_suffix('.records.json'),rows)
    meta=dict(groups=len(rows),repeats=repeats,continuations=count,continuation_plies=plies,seconds=time.perf_counter()-start,sha256=sha(path),source_hashes=source_hashes(),baseline_sha256=sha(BASE),workers=workers)
    write(path.with_suffix('.json'),meta);return meta
