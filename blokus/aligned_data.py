"""Search-aligned root/reply groups and per-ply common-random continuations."""
import concurrent.futures
import multiprocessing
import random
import time
from pathlib import Path
import numpy as np
from .engine import State,legal_moves,play
from .ai import seed_all,features,heuristic
from .research import clone,sha,source_hashes
from .value_search import ValueEvaluator,encode,SCALE
from .value_experiment import write

BASE='artifacts/value-search/compact/seed-201/predictive-best.pt'
POOL=None


class FixedValue:
    """Deterministic fixed-width version of the incumbent's leaf evaluator."""
    def __init__(self,learned=True):
        self.evaluator=ValueEvaluator(BASE if learned else None,'cpu',.5,24)
    def act(self,s,rng):
        qs=legal_moves(s)
        if not qs:return None
        x,_=features(s,qs);order=np.argsort(-heuristic(x),kind='stable')[:4];leaves=[];ranges=[]
        for i in order:
            child=clone(s);play(child,qs[int(i)].move,validate=False);rs=legal_moves(child);begin=len(leaves)
            if rs:
                rx,_=features(child,rs)
                for j in np.argsort(-heuristic(rx),kind='stable')[:3]:
                    leaf=clone(child);play(leaf,rs[int(j)].move,validate=False);leaves.append(leaf)
            else:
                play(child,None,validate=False);leaves.append(child)
            ranges.append((begin,len(leaves)))
        v=self.evaluator.batch(leaves);values=np.array([v[a:b].min() for a,b in ranges]);ties=np.flatnonzero(values>=values.max()-1e-7)
        return qs[int(order[int(rng.choice(ties))])].move


def initialize():
    global POOL
    seed_all(555);POOL=[FixedValue(False),FixedValue(True)]


def ply_seed(seed,ply,seat):
    # Fixed integer mix independent of process hash salts and random consumption.
    return (seed*6364136223846793005+ply*1442695040888963407+seat*3202034522624059733)&((1<<64)-1)


def finish(leaf,seed,condition):
    s=clone(leaf);perspective=s.turn;ply=0;patterns=[(0,0),(1,1),(1,0),(0,1)];pattern=patterns[condition%4]
    while not s.done:
        rng=random.Random(ply_seed(seed,ply,s.turn));qs=legal_moves(s)
        if qs and rng.random()<.05:m=rng.choice(qs).move
        else:m=POOL[pattern[0 if s.turn==perspective else 1]].act(s,rng)
        play(s,m,validate=False);ply+=1
    return (s.score(perspective)-s.score(1-perspective))/SCALE,ply


def replies_group(root):
    qs=legal_moves(root)
    if len(qs)<4:return None
    x,_=features(root,qs);order=np.argsort(-heuristic(x),kind='stable')[:4];leaves=[];moves=[];replies=[]
    for i in order:
        child=clone(root);m=qs[int(i)].move;play(child,m,validate=False);rs=legal_moves(child);branch=[];rm=[]
        if rs:
            rx,_=features(child,rs)
            for j in np.argsort(-heuristic(rx),kind='stable')[:3]:
                leaf=clone(child);reply=rs[int(j)].move;play(leaf,reply,validate=False);branch.append(leaf);rm.append([reply.piece,reply.orientation,reply.row,reply.col])
        else:
            play(child,None,validate=False);branch=[child];rm=[None]
        leaves.append(branch);moves.append([m.piece,m.orientation,m.row,m.col]);replies.append(rm)
    return leaves,moves,replies


def generate(task):
    seed,groups,repeats=task;rng=random.Random(seed);s=State();prefix=[]
    for _ in range(rng.randint(2,6)):
        m=rng.choice(legal_moves(s)).move;prefix.append([m.piece,m.orientation,m.row,m.col]);play(s,m)
    trace=[]
    # Root distribution includes the incumbent, not just handmade policies.
    while not s.done:
        qs=legal_moves(s)
        if sum(r.bit_count() for r in s.remaining)<=24 and len(qs)>=4:trace.append(clone(s))
        m=rng.choice(qs).move if qs and rng.random()<.05 else POOL[seed%2].act(s,rng)
        play(s,m,validate=False)
    output=[];meta=[];plies=0;continuations=0
    for g,root in enumerate(rng.sample(trace,min(groups,len(trace)))):
        leaves,moves,replies=replies_group(root);b=np.zeros((4,3,147),dtype=np.uint8);x=np.zeros((4,3,49),dtype=np.float32)
        ys=np.zeros((4,3,repeats),dtype=np.float32);mask=np.zeros((4,3),dtype=bool);seeds=[seed*1000+g*100+r for r in range(repeats)]
        for k,branch in enumerate(leaves):
            for j,leaf in enumerate(branch):
                eb,ex=encode(leaf);b[k,j]=np.packbits(eb.reshape(-1));x[k,j]=ex;mask[k,j]=True
                for r,rs in enumerate(seeds):ys[k,j,r],n=finish(leaf,rs,r);plies+=n;continuations+=1
        output.append((b,x,ys,mask));meta.append(dict(game=seed,root=dict(boards=root.boards,remaining=root.remaining,last=root.last,turn=root.turn,passes=root.passes),moves=moves,replies=replies,continuation_seeds=seeds))
    return output,meta,prefix,plies,continuations


def collect(path,games=600,seed=300000,groups=2,repeats=4,workers=6):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise ValueError('dataset exists')
    begin=time.perf_counter();bs=[];xs=[];ys=[];ms=[];ids=[];metadata=[];prefixes=[];plies=0;count=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers,initializer=initialize,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,(rows,met,prefix,n,c) in enumerate(pool.map(generate,[(seed+i,groups,repeats) for i in range(games)],chunksize=2)):
            for b,x,y,m in rows:bs.append(b);xs.append(x);ys.append(y);ms.append(m);ids.append(seed+i)
            metadata.extend(met);prefixes.append(prefix);plies+=n;count+=c
            if (i+1)%20==0:print('generated',i+1,len(ids),round(time.perf_counter()-begin,2),'seconds',flush=True)
    np.savez_compressed(path,board=bs,state=xs,returns=ys,mask=ms,game=ids)
    write(path.with_suffix('.groups.json'),metadata);write(path.with_suffix('.prefixes.json'),prefixes)
    summary=dict(games=games,groups=len(ids),repeats=repeats,continuations=count,continuation_plies=plies,seconds=time.perf_counter()-begin,sha256=sha(path),source_hashes=source_hashes(),baseline_sha256=sha(BASE),workers=workers,seed=seed)
    write(path.with_suffix('.json'),summary);print(summary,flush=True);return summary


def quality(path):
    d=np.load(path);y=d['returns'];mask=d['mask'];ref=np.where(mask,y[:,:, :,8:12].mean(-1),np.inf).min(-1)
    result={}
    for n in [2,4,8]:
        v=np.where(mask,y[:,:,:,:n].mean(-1),np.inf).min(-1);choice=v.argmax(-1);best=ref.argmax(-1)
        result[n]=dict(candidate_mse=float(np.square(v-ref).mean()),decision_regret_points=float((ref.max(-1)-ref[np.arange(len(v)),choice]).mean()*40),top_agreement=float((choice==best).mean()))
    # Full cycles cover all four policy seat assignments. Use 8 if it improves
    # both average decision regret and candidate MSE against held-out repeats.
    repeats=8 if result[8]['candidate_mse']<result[4]['candidate_mse'] and result[8]['decision_regret_points']<result[4]['decision_regret_points'] else 4
    report=dict(samples=len(ref),against='disjoint repeats 8..11; noisy reference, not ground truth',metrics=result,selected_training_repeats=repeats)
    write(Path(path).with_suffix('.quality.json'),report);return report
