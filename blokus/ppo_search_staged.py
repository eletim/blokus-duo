"""Completed narrow 4ply first, optional same-depth nested root widening. WDL units."""
import copy,time
import torch
from .engine import legal_moves,play
from .ppo_policy import collate
from .ppo_value_leaf import value_only

def key(s):return (tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)
def child(s,m):
    c=copy.deepcopy(s);play(c,m,validate=False);return c

class ValueLeafSearch:
    def __init__(self,policy,budget_ms=500,width=4,max_depth=4,root_stages=(),reuse=False):
        if not 10<=budget_ms<=5000 or width<1 or max_depth not in (2,4):raise ValueError('invalid limits')
        if any(n<=width for n in root_stages) or tuple(sorted(set(root_stages)))!=tuple(root_stages):raise ValueError('root stages must strictly increase')
        self.policy=policy;self.budget_ms=budget_ms;self.width=width;self.max_depth=max_depth;self.root_stages=tuple(root_stages);self.reuse=reuse;self.stats={}
    def act(self,state,rng):
        start=time.perf_counter();deadline=start+self.budget_ms/1000;policies={};values={};memo={}
        stats=dict(depth=0,root_width=0,nodes=0,nn_policy_calls=0,nn_value_calls=0,cache_hits=0,stages=[],fallback=True,forced_pass=False)
        def check():
            if time.perf_counter()>=deadline:raise TimeoutError
        def evaluate(s,leaf=False):
            check();stats['nodes']+=1;k=key(s)
            if s.done:return [],value_only(self.policy,s)
            if leaf:
                if self.reuse and k in values:stats['cache_hits']+=1;return [],values[k]
                stats['nn_value_calls']+=1;v=value_only(self.policy,s);check()
                if self.reuse:values[k]=v
                return [],v
            if self.reuse and k in policies:stats['cache_hits']+=1;return policies[k]
            obs,qs=self.policy._observation(s)
            with torch.inference_mode():
                logits,v=self.policy.model(*collate([obs],self.policy.device));order=logits[0].argsort(descending=True).cpu().tolist();v=v.item()
            stats['nn_policy_calls']+=1;check();result=([qs[i].move for i in order] if qs else [None],v)
            if self.reuse:policies[k]=result;values[k]=v
            return result
        def visit(s,depth):
            check();k=(key(s),depth,self.width)
            if self.reuse and k in memo:stats['cache_hits']+=1;return memo[k]
            moves,v=evaluate(s,not depth)
            if depth and not s.done:v=max(-visit(child(s,m),depth-1) for m in moves[:self.width])
            if self.reuse:memo[k]=v
            return v
        selected=self.policy.act(state,rng);baseline=selected
        if not legal_moves(state):
            stats.update(forced_pass=True,elapsed_ms=(time.perf_counter()-start)*1000);self.stats=stats;return None
        try:
            order,_=evaluate(state);base=order[:self.width]
            if baseline not in base:base[-1]=baseline
            nested=base+[m for m in order if m not in base]
            stages=[(2,self.width)]
            if self.max_depth==4:stages.append((4,self.width));stages.extend((4,n) for n in self.root_stages)
            for depth,width in stages:
                roots=nested[:width];scores=[-visit(child(state,m),depth-1) for m in roots];check()
                selected=roots[max(range(len(scores)),key=scores.__getitem__)];stats.update(depth=depth,root_width=len(roots),root_scores=scores,fallback=False)
                stats['stages'].append(dict(depth=depth,width=len(roots)))
        except TimeoutError:pass
        stats['elapsed_ms']=(time.perf_counter()-start)*1000;self.stats=stats;return selected
