"""Fixed-value selective iterative deepening and separately certified full endgames."""
from dataclasses import dataclass
import time
import numpy as np
from .engine import legal_moves,play
from .research import clone
from .ai import features,heuristic
from .value_search import BudgetAgent,ValueEvaluator,SCALE

EXACT,LOWER,UPPER='EXACT','LOWER','UPPER'
class SearchTimeout(Exception):pass


def key(s):
    return (*s.boards,*s.remaining,s.score(0),s.score(1),s.turn,s.passes)


def score(s):return float(s.score(s.turn)-s.score(1-s.turn))


@dataclass
class Entry:
    value:float
    bound:str
    move:object=None


class TreeSearch:
    def __init__(self,evaluator,width=3,clock=time.perf_counter,tt_limit=100000):
        self.evaluator=evaluator;self.width=width;self.clock=clock;self.deadline=float('inf');self.tt={};self.tt_limit=tt_limit
        self.nodes=0;self.hits=0;self.cutoffs=0;self.leaves=0;self.max_ply=0
    def check(self):
        if self.clock()>=self.deadline:raise SearchTimeout()
    def ordered(self,s,qs,full=False):
        if not qs:return [None]
        x,_=features(s,qs);order=np.argsort(-heuristic(x),kind='stable')
        if not full:order=order[:self.width]
        return [qs[int(i)].move for i in order]
    def put(self,k,entry):
        if len(self.tt)>=self.tt_limit:self.tt.clear()
        self.tt[k]=entry
    def search(self,s,depth,alpha=-float('inf'),beta=float('inf'),full=False,ply=0):
        self.check();self.nodes+=1;self.max_ply=max(self.max_ply,ply)
        if s.done or s.remaining==[0,0]:return score(s)
        k=(key(s),-1 if full else depth);a0,b0=alpha,beta;entry=self.tt.get(k)
        if entry:
            self.hits+=1
            if entry.bound==EXACT:return entry.value
            if entry.bound==LOWER:alpha=max(alpha,entry.value)
            else:beta=min(beta,entry.value)
            if alpha>=beta:self.cutoffs+=1;return entry.value
        if not full and depth<=0:
            v=float(self.evaluator.batch([s])[0])*SCALE;self.check();self.leaves+=1
            self.put(k,Entry(v,EXACT));return v
        qs=legal_moves(s);self.check()
        if not qs and s.passes:return score(s)
        moves=self.ordered(s,qs,full)
        # Only reorder the fixed shortlist. A TT move never changes its membership.
        if entry and entry.move in moves:moves.remove(entry.move);moves.insert(0,entry.move)
        best=-float('inf');best_move=None
        if depth==1 and not full:
            children=[]
            for m in moves:
                child=clone(s);play(child,m,validate=False);children.append(child)
            self.check();values=-self.evaluator.batch(children)*SCALE;self.check()
            self.nodes+=len(children);self.leaves+=len(children);self.max_ply=max(self.max_ply,ply+1)
            i=int(np.argmax(values));best=float(values[i]);best_move=moves[i]
        else:
            for m in moves:
                child=clone(s);play(child,m,validate=False)
                v=-self.search(child,depth-1,-beta,-alpha,full,ply+1)
                if v>best:best,best_move=v,m
                alpha=max(alpha,best)
                if alpha>=beta:self.cutoffs+=1;break
        self.check()
        bound=UPPER if best<=a0 else LOWER if best>=b0 else EXACT
        self.put(k,Entry(best,bound,best_move));return best
    def root_two_ply(self,s,moves,rng):
        """Complete all roots with the old three-reply rule and old tie RNG."""
        values=[]
        for m in moves:
            self.check();child=clone(s);play(child,m,validate=False)
            values.append(-self.search(child,1))
        self.check();v=np.asarray(values);best=np.flatnonzero(v>=v.max()-1e-7*SCALE)
        return float(v.max()),moves[int(rng.choice(best))]
    def root(self,s,moves,depth,full=False,preferred=None):
        order=list(moves)
        if preferred in order:order.remove(preferred);order.insert(0,preferred)
        best=-float('inf');choice=None
        for m in order:
            self.check();child=clone(s);play(child,m,validate=False)
            v=-self.search(child,depth-1,-float('inf'),-best,full,1)
            # Equal fail-soft bounds are not proof of a tied optimum.
            if v>best:best,choice=v,m
        self.check();return best,choice


class DeepAgent:
    def __init__(self,checkpoint,budget_ms=25,root_width=8,width=3,deep_gate=24,exact_pieces=8,exact_root_limit=32,max_depth=10,device='cuda',exact_mobility=0,only_exact=False):
        if budget_ms<=0 or root_width<1 or width<1:raise ValueError('positive budget and widths required')
        self.budget_ms=budget_ms;self.root_width=root_width;self.deep_gate=deep_gate;self.exact_pieces=exact_pieces;self.exact_root_limit=exact_root_limit;self.max_depth=max_depth;self.exact_mobility=exact_mobility;self.only_exact=only_exact
        self.old=BudgetAgent('value',checkpoint,.5,24,budget_ms,device);self.tree=TreeSearch(self.old.evaluator,width)
        self.times=[];self.decisions=[];self.exact_attempts=0;self.exact_solved=0;self.aborts=0;self.fallbacks=0
    def act(self,s,rng):
        if s.done:raise ValueError('terminal state')
        start=time.perf_counter();deadline=start+self.budget_ms/1000;before=self.tree.nodes;old_before=self.old.nodes
        remaining=sum(v.bit_count() for v in s.remaining);depth=0;solved=False;value=None;phase='selective';attempt=False;eligible=False;mobility=None
        if remaining>self.deep_gate:
            phase='old-opening';self.old.budget_ms=max(.01,(deadline-time.perf_counter())*1000);move=self.old.act(s,rng)
        else:
            # Retain exact entries across turns; selective entries are depth-keyed.
            self.tree.deadline=deadline-max(.00015,self.budget_ms/1000*.015)
            qs=legal_moves(s)
            if self.exact_mobility and qs:
                opponent=clone(s);opponent.turn=1-s.turn
                mobility=len(qs)+len(legal_moves(opponent))
            eligible=bool(qs) and len(qs)<=self.exact_root_limit and ((bool(self.exact_pieces) and remaining<=self.exact_pieces) or (mobility is not None and mobility<=self.exact_mobility))
            if not qs:move=None;phase='forced-pass'
            elif self.only_exact and not eligible:
                phase='old-mobility';self.old.budget_ms=max(.01,(deadline-time.perf_counter())*1000);move=self.old.act(s,rng)
            else:
                x,_=features(s,qs);rank=np.argsort(-heuristic(x),kind='stable');all_moves=[qs[int(i)].move for i in rank]
                shortlist=all_moves if self.only_exact else all_moves[:self.root_width];move=shortlist[0]
                try:
                    for d in range(2,self.max_depth+1,2):
                        if self.only_exact:v,m=self.tree.root_two_ply(s,shortlist,rng)
                        else:v,m=self.tree.root(s,shortlist,d,preferred=move)
                        if time.perf_counter()>=self.tree.deadline:raise SearchTimeout()
                        value,move,depth=v,m,d
                        if d==2 and eligible:
                            attempt=True;self.exact_attempts+=1
                            v,m=self.tree.root(s,all_moves,0,full=True,preferred=move)
                            if time.perf_counter()>=self.tree.deadline:raise SearchTimeout()
                            value,move,solved=v,m,True;self.exact_solved+=1;phase='exact';break
                        if self.only_exact:break
                except SearchTimeout:self.aborts+=1
                if depth==0:self.fallbacks+=1
        elapsed=time.perf_counter()-start;self.times.append(elapsed)
        self.decisions.append(dict(phase=phase,completed_depth=depth,exact=solved,exact_attempt=attempt,exact_eligible=eligible,mobility=mobility,value=value,nodes=self.tree.nodes-before+self.old.nodes-old_before,ms=elapsed*1000))
        return move
    def stats(self):
        t=np.array(self.times)*1000;active=[x for x in self.decisions if not x['phase'].startswith('old-') and x['phase']!='forced-pass']
        return dict(calls=len(t),milliseconds_per_move=float(t.mean()) if len(t) else 0,p50_ms=float(np.median(t)) if len(t) else 0,p95_ms=float(np.quantile(t,.95)) if len(t) else 0,max_ms=float(t.max()) if len(t) else 0,
                    total_seconds=float(t.sum()/1000),allowance_ms=self.budget_ms,overruns=int((t>self.budget_ms).sum()),overrun_ms=float(np.maximum(t-self.budget_ms,0).sum()),nodes=sum(x['nodes'] for x in self.decisions),
                    selective_calls=len(active),completed_depth_mean=float(np.mean([x['completed_depth'] for x in active])) if active else 0,depth_histogram={str(d):sum(x['completed_depth']==d for x in active) for d in range(0,self.max_depth+1,2)},
                    exact_attempts=self.exact_attempts,exact_solved=self.exact_solved,exact_fraction_all_calls=self.exact_solved/len(t) if len(t) else 0,exact_fraction_attempts=self.exact_solved/self.exact_attempts if self.exact_attempts else 0,tt_hits=self.tree.hits,cutoffs=self.tree.cutoffs,aborted_iterations=self.aborts,fallbacks=self.fallbacks,old_opening_calls=sum(x['phase']=='old-opening' for x in self.decisions),old_mobility_calls=sum(x['phase']=='old-mobility' for x in self.decisions))
