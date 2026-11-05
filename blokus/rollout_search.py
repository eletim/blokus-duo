"""Terminal score rollouts, common random numbers, uniform/halving allocation."""
import math,random,time
import numpy as np
from .engine import legal_moves,play
from .ai import features,heuristic
from .research import clone
from .node_diagnostic import row

class RolloutTimeout(Exception):pass

def check(deadline,clock):
    if clock()>=deadline:raise RolloutTimeout()

def candidates(s,count=8,diverse=True):
    qs=legal_moves(s)
    if not qs:return [None]
    x,_=features(s,qs);rank=[qs[int(i)].move for i in np.argsort(-heuristic(x),kind='stable')]
    if not diverse:return rank[:count]
    chosen=rank[:min(4,count)];seen={(m.piece,m.row//7,m.col//7) for m in chosen}
    for m in rank[4:]:
        sig=(m.piece,m.row//7,m.col//7)
        if len(chosen)>=count:break
        if sig not in seen:chosen.append(m);seen.add(sig)
    for m in rank:
        if len(chosen)>=count:break
        if m not in chosen:chosen.append(m)
    return chosen

def terminal_rollout(s,move,seed,deadline=float('inf'),clock=time.perf_counter,trace=False):
    check(deadline,clock);p=s.turn;c=clone(s);play(c,move,validate=False);rng=random.Random(seed);moves=[row(move)];plies=1
    while not c.done:
        check(deadline,clock);u=rng.random() # Also consumed on pass, keyed by continuation ply.
        qs=legal_moves(c);check(deadline,clock)
        if qs:
            x,_=features(c,qs);idx=np.argsort(-heuristic(x),kind='stable')[:3]
            m=qs[int(idx[min(int(u*len(idx)),len(idx)-1)])].move
        else:m=None
        check(deadline,clock);play(c,m,validate=False);plies+=1
        if trace:moves.append(row(m))
        if plies>86:raise AssertionError('rollout did not terminate')
    check(deadline,clock)
    return dict(value=c.score(p)-c.score(1-p),plies=plies,moves=moves if trace else None,scores=[c.score(0),c.score(1)])


def allocate(s,moves,seed=0,total=32,method='halving',deadline=float('inf'),clock=time.perf_counter,trace=False,shared=True):
    if method not in ('uniform','halving'):raise ValueError('method')
    if total<len(moves):raise ValueError('budget must cover each candidate')
    active=list(range(len(moves)));values=[[] for _ in moves];labels=[];committed=0;completed=0;attempted=0;discarded=0;choice=0;stage=0;interrupted=False;rounds=0
    # Stage quotas preserve the specified total bound; for8,total32:16+8+8.
    stages=max(1,math.ceil(math.log2(len(moves)))) if method=='halving' else 1
    while active and completed<total:
        remaining_stages=max(1,stages-stage)
        quota=(total-completed+1)//2 if remaining_stages>1 else total-completed
        reps=quota//len(active)
        if reps<1:break
        for _ in range(reps):
            pending=[]
            try:
                for i in active:
                    sample=len(values[i]);sample_seed=seed+sample*1000003+(0 if shared else i*10007)
                    attempted+=1
                    z=terminal_rollout(s,moves[i],sample_seed,deadline,clock,trace)
                    completed+=1;pending.append((i,z));labels.append(dict(candidate=i,sample=sample,seed=sample_seed,**z))
                check(deadline,clock)
            except RolloutTimeout:
                discarded+=len(pending);interrupted=True;break
            for i,z in pending:values[i].append(z['value'])
            committed+=len(pending);rounds+=1
            choice=max(active,key=lambda i:(np.mean(values[i]),-i))
        if interrupted:break
        if method=='uniform' or len(active)==1:break
        active=sorted(active,key=lambda i:(-np.mean(values[i]),i))[:math.ceil(len(active)/2)];stage+=1
        if len(active)==1:break
    return dict(move=row(moves[choice]),candidate=choice,completed=completed,committed=committed,discarded_completed=discarded,attempted=attempted,interrupted=interrupted,fallback=not all(values),rounds=rounds,stages=stage,values=values,labels=labels)

class RolloutAgent:
    def __init__(self,budget_ms=25,method='halving',diverse=True,count=8,seed=900):
        self.budget_ms=budget_ms;self.method=method;self.diverse=diverse;self.count=count;self.seed=seed;self.turn=0;self.times=[];self.decisions=[]
    def reset_game(self,seed):self.seed=seed;self.turn=0
    def act(self,s,rng):
        if s.done:raise ValueError('terminal')
        start=time.perf_counter();deadline=start+self.budget_ms/1000;ms=candidates(s,self.count,self.diverse)
        if ms==[None]:z=dict(candidate=0,completed=0,committed=0,discarded_completed=0,fallback=False,interrupted=False,stages=0,rounds=0)
        else:
            # Fixed32 sample plan: enough data to test adaptive vs uniform allocation.
            z=allocate(s,ms,self.seed+self.turn*20000003,32,self.method,deadline)
        self.turn+=1;elapsed=time.perf_counter()-start;self.times.append(elapsed)
        self.decisions.append({k:z[k] for k in ['completed','committed','discarded_completed','fallback','interrupted','stages','rounds']}|dict(ms=elapsed*1000,candidates=len(ms)))
        return ms[z['candidate']]
    def stats(self):
        t=np.array(self.times)*1000
        return dict(calls=len(t),mean_ms=float(t.mean()),p95_ms=float(np.quantile(t,.95)),max_ms=float(t.max()),overruns=int((t>self.budget_ms).sum()),completed=sum(z['completed'] for z in self.decisions),committed=sum(z['committed'] for z in self.decisions),fallbacks=sum(z['fallback'] for z in self.decisions))
