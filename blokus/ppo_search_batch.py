"""Research-only completed beam negamax; WDL units, cold per decision cache."""
import copy
import time
import torch
from .engine import legal_moves, play
from .ppo_policy import collate

def key(s):
    return (tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)
def terminal(s):
    d=s.score(s.turn)-s.score(1-s.turn)
    return float((d>0)-(d<0))
def child(s,m):
    c=copy.deepcopy(s);play(c,m,validate=False);return c

class BatchSearch:
    def __init__(self,policy,budget_ms=500,root_width=12,width=4,max_depth=4,exact=False):
        if budget_ms<=0 or min(root_width,width,max_depth)<1:raise ValueError('invalid limits')
        self.policy=policy;self.budget_ms=budget_ms;self.root_width=root_width;self.width=width;self.max_depth=max_depth;self.exact=exact;self.stats={}
    def act(self,s,rng):
        start=time.perf_counter();deadline=start+self.budget_ms/1000;cache={};batch_cost=.002
        stats=dict(depth=0,nodes=0,nn_states=0,nn_calls=0,exact_nodes=0,exact_solved=False,fallback=True,completed_roots=0,observation_ms=0.,nn_ms=0.)
        def check(cost=0.):
            if time.perf_counter()+max(.001,cost)>deadline:raise TimeoutError
        def evaluate(states):
            nonlocal batch_cost
            pending=[];seen=set()
            for z in states:
                k=key(z)
                if k in cache or k in seen:continue
                seen.add(k);stats['nodes']+=1
                if z.done:cache[k]=([],terminal(z));continue
                pending.append(z)
            for off in range(0,len(pending),16):
                check(batch_cost);t=time.perf_counter();chunk=pending[off:off+16];obs=[];qs=[]
                for z in chunk:
                    check();o,q=self.policy._observation(z);obs.append(o);qs.append(q)
                stats['observation_ms']+=(time.perf_counter()-t)*1000;t1=time.perf_counter();check()
                with torch.inference_mode():
                    logits,values=self.policy.model(*collate(obs,self.policy.device))
                    orders=logits.argsort(dim=1,descending=True).cpu().tolist();values=values.cpu().tolist()
                stats['nn_ms']+=(time.perf_counter()-t1)*1000;stats['nn_calls']+=1;stats['nn_states']+=len(chunk)
                batch_cost=max(.001,time.perf_counter()-t)
                for z,q,order,v in zip(chunk,qs,orders,values):cache[key(z)]=([q[i].move for i in order if i<len(q)] if q else [None],v)
                check()
        # Policy fallback remains valid even if mandatory inference overruns budget.
        selected=self.policy.act(s,rng);base=selected
        try:
            evaluate([s]);roots=cache[key(s)][0][:self.root_width]
            if selected not in roots:roots[-1]=selected
            for depth in range(2,self.max_depth+1,2):
                layers=[[child(s,m) for m in roots]];edges=[]
                for level in range(1,depth+1):
                    evaluate(layers[-1])
                    if level==depth:break
                    nxt=[];links=[]
                    for z in layers[-1]:
                        check();indices=[]
                        if not z.done:
                            for m in cache[key(z)][0][:self.width]:indices.append(len(nxt));nxt.append(child(z,m))
                        links.append(indices)
                    edges.append(links);layers.append(nxt)
                vals=[cache[key(z)][1] for z in layers[-1]]
                for level in range(len(edges)-1,-1,-1):
                    vals=[max(-vals[j] for j in js) if js else cache[key(layers[level][i])][1] for i,js in enumerate(edges[level])]
                check();scores=[-v for v in vals];best=max(range(len(scores)),key=scores.__getitem__);selected=roots[best]
                stats.update(depth=depth,fallback=False,completed_roots=len(roots),root_scores=scores)
            if self.exact and sum(r.bit_count() for r in s.remaining)<=6:
                memo={}
                def solve(z):
                    check();k=key(z)
                    if k in memo:return memo[k]
                    stats['exact_nodes']+=1
                    if z.done:v=terminal(z)
                    else:v=max(-solve(child(z,q.move if q else None)) for q in (legal_moves(z) or [None]))
                    memo[k]=v;return v
                moves=[q.move for q in legal_moves(s)] or [None];scores=[-solve(child(s,m)) for m in moves];check()
                selected=moves[max(range(len(scores)),key=scores.__getitem__)];stats.update(exact_solved=True,fallback=False,completed_roots=len(moves))
        except TimeoutError:pass
        stats.update(elapsed_ms=(time.perf_counter()-start)*1000,changed=selected!=base)
        self.stats=stats;return selected
