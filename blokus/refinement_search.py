"""Atomic root refinement after a broad two-ply baseline; selective negamax.

Every value is in the root player's perspective. Additional depth replaces a
root value only after its entire search completes. Unknown roots never compete.
No TT: cold games/phases are independent. This is not full-width minimax.
"""
import time
import numpy as np
from .value_search import BudgetAgent
from .deep_search import TreeSearch, SearchTimeout
from .engine import legal_moves, play
from .research import clone
from .ai import features, heuristic

class NodeLimit(SearchTimeout): pass

class LimitedTree(TreeSearch):
    def __init__(self,*args,node_limit=10**9,**kwargs):
        super().__init__(*args,**kwargs);self.node_limit=node_limit
    def put(self,k,entry):
        pass # Deliberately no cache, including within a decision.
    def search(self,s,depth,*args,**kwargs):
        need=4 if depth==1 and not kwargs.get("full",False) else 1
        if self.nodes+need>self.node_limit:raise NodeLimit()
        return super().search(s,depth,*args,**kwargs)
    def check(self):
        super().check()
        if self.nodes>=self.node_limit:raise NodeLimit()

class RefinementAgent(BudgetAgent):
    def __init__(self,checkpoint=None,budget_ms=25,method='priority',device='cuda',node_limit=None,reserve_fraction=0.):
        super().__init__('value',checkpoint,.5,24,budget_ms,device)
        if method not in ('priority','uniform','control'):raise ValueError(method)
        self.method=method;self.node_limit=node_limit;self.reserve_fraction=reserve_fraction
        self.decisions=[];self.reset_game()
    def reset_game(self):
        self.cost_estimate=.0008;self.refine_estimate=.008
    def act(self,s,rng):
        if s.done:raise ValueError('terminal')
        start=time.perf_counter();margin=max(.001,self.budget_ms*.00005)
        deadline=start+self.budget_ms/1000-margin if self.node_limit is None else float('inf')
        initial_deadline=deadline-self.budget_ms/1000*self.reserve_fraction if self.node_limit is None else deadline
        qs=legal_moves(s);t=time.perf_counter();order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128] if qs else []
        rank_end=time.perf_counter();values={};tree=LimitedTree(self.evaluator,width=3,node_limit=self.node_limit or 10**9)
        tree.deadline=initial_deadline;tree.tt_limit=0
        for i in order:
            if time.perf_counter()+1.5*self.cost_estimate>=initial_deadline:break
            step=time.perf_counter();c=clone(s);play(c,qs[int(i)].move,validate=False)
            try:v=-tree.search(c,1)
            except SearchTimeout:break
            self.cost_estimate=max(.00015,.7*self.cost_estimate+.3*min(time.perf_counter()-step,self.budget_ms/1000))
            values[int(i)]=v
        initial_end=time.perf_counter();initial=dict(values);refined=[];attempted=0
        # No artificial narrow candidate set in v1. Revised reserve variant may
        # refine only fully evaluated roots; this is explicitly a separate control.
        eligible=len(values)==len(order) or self.reserve_fraction>0
        tree.deadline=deadline
        if values and eligible and self.method!='control':
            sequence=sorted(values,key=lambda i:(-values[i],list(order).index(i))) if self.method=='priority' else list(values)
            for i in sequence:
                if time.perf_counter()+1.5*self.refine_estimate>=deadline:break
                attempted+=1;step=time.perf_counter();c=clone(s);play(c,qs[i].move,validate=False)
                try:v=-tree.search(c,3)
                except SearchTimeout:break
                self.refine_estimate=max(.0003,.7*self.refine_estimate+.3*min(time.perf_counter()-step,self.budget_ms/1000))
                values[i]=v;refined.append(i)
        if values:
            best=max(values.values());choice=int(rng.choice([i for i,v in values.items() if v>=best-4e-6]));move=qs[choice].move
        else:move=qs[int(order[0])].move if len(order) else None;self.fallbacks+=bool(len(order))
        elapsed=time.perf_counter()-start;self.times.append(elapsed);self.nodes+=tree.nodes;self.roots+=len(initial)
        self.decisions.append(dict(ms=elapsed*1000,nodes=tree.nodes,completed_roots=len(initial),candidate_count=len(order),refined=len(refined),attempted=attempted,
            fallback=bool(len(order) and not values),baseline_retained=not refined,initial_complete=len(initial)==len(order),
            sections_ms=dict(legal=(t-start)*1000,rank=(rank_end-t)*1000,initial=(initial_end-rank_end)*1000,extra=(time.perf_counter()-initial_end)*1000)))
        self.last_values=values;self.initial_values=initial
        return move
    def stats(self):
        result=super().stats();result.pop('sections')
        result['sections_ms']={k:sum(d['sections_ms'][k] for d in self.decisions) for k in ('legal','rank','initial','extra')}
        result['completed_refinements']=sum(d['refined'] for d in self.decisions)
        result['attempted_refinements']=sum(d['attempted'] for d in self.decisions)
        result['baseline_retained_calls']=sum(d['baseline_retained'] for d in self.decisions)
        return result
