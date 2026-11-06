"""Completed-stage selective negamax; frozen non-neural policy and WDL value."""
import copy,time
import numpy as np
from .engine import play
from .linear_value import Search,outcome

class CompletedSearch(Search):
 def __init__(self,agent,ms=25,width=6,replies=3,depth=2,base_width=6,base_replies=3,node_limit=None):
  super().__init__(agent,ms,width,replies,depth)
  if depth not in (2,4):raise ValueError('completed search depth must be 2 or 4')
  for n in (base_width,base_replies):
   if n is not None and (not isinstance(n,int) or n<1):raise ValueError('invalid base width')
  self.base_width=base_width;self.base_replies=base_replies;self.node_limit=node_limit
 def act(self,s,rng):
  begin=time.perf_counter();deadline=begin+self.ms/1000 if self.ms else float('inf')
  nodes=0;input_s=0.;value_s=0.;input_max=.0002;value_max=.0002;reply_count=0
  def guard(cost=0.):
   if time.perf_counter()+cost+.0002>=deadline:raise TimeoutError
  def inputs(t):
   nonlocal input_s,input_max
   q=time.perf_counter();r=self.agent.inputs(t);dt=time.perf_counter()-q;input_s+=dt;input_max=max(input_max,dt);return r
  x,qs,p=inputs(s)
  if not qs:
   self.stats.append(dict(ms=(time.perf_counter()-begin)*1000,forced_pass=True,complete=True,fallback=False,depth=0,nodes=0));return None
  scores=x@self.agent.weights[p];baseline=qs[int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)))].move
  ordered=[qs[i].move for i in np.argsort(-scores,kind='stable')]
  def roots(width):
   r=ordered[:width] if width else ordered[:]
   if baseline not in r:r[-1]=baseline
   assert len(set(r))==len(r)
   return r
  def visit(t,d,replies):
   nonlocal nodes,value_s,value_max,reply_count
   guard()
   if self.node_limit is not None and nodes>=self.node_limit:raise TimeoutError
   nodes+=1
   if t.done:return outcome(t,t.turn)
   if d==0:
    guard(value_max);q=time.perf_counter();v=self.agent.value(t);dt=time.perf_counter()-q;value_s+=dt;value_max=max(value_max,dt);return v
   guard(input_max);a,c,k=inputs(t)
   ids=np.argsort(-(a@self.agent.weights[k]),kind='stable') if c else []
   moves=[c[i].move for i in (ids[:replies] if replies else ids)] if c else [None]
   best=-float('inf')
   for m in moves:
    guard();child=copy.deepcopy(t);play(child,m,validate=False);reply_count+=1;best=max(best,-visit(child,d-1,replies))
   return best
  stages=[(self.base_width,self.base_replies,2)]
  if (self.width,self.replies,self.depth)!=stages[0]:stages.append((self.width,self.replies,self.depth))
  completed=[];attempts=[];move=baseline;chosen_values=[];chosen_roots=[]
  for width,replies,depth in stages:
   rr=roots(width);values=[];ok=False
   try:
    for m in rr:
     guard();child=copy.deepcopy(s);play(child,m,validate=False);values.append(-visit(child,depth-1,replies))
    if time.perf_counter()>deadline:raise TimeoutError
    ok=True
   except TimeoutError:pass
   attempts.append(dict(depth=depth,width=width,replies=replies,roots=len(rr),completed_roots=len(values),complete=ok))
   if not ok:break
   # Consume a selection draw only once, after final completed stage.
   completed.append(attempts[-1]);chosen_values=values;chosen_roots=rr
  if completed:move=chosen_roots[int(rng.choice(np.flatnonzero(np.asarray(chosen_values)>=max(chosen_values)-1e-12)))]
  elapsed=(time.perf_counter()-begin)*1000;last=completed[-1] if completed else {}
  self.stats.append(dict(ms=elapsed,nodes=nodes,fallback=not completed,complete=bool(completed),depth=last.get('depth',0),root_width=last.get('width'),reply_width=last.get('replies'),root_candidates=len(chosen_roots),completed_roots=len(chosen_values),legal_candidates=len(qs),values=chosen_values,roots=[[m.piece,m.orientation,m.row,m.col] for m in chosen_roots],attempts=attempts,target_complete=len(completed)==len(stages),overshoot=self.ms>0 and elapsed>self.ms,overshoot_ms=max(0.,elapsed-self.ms) if self.ms else 0.,input_ms=input_s*1000,value_ms=value_s*1000,replies_evaluated=reply_count,baseline=[baseline.piece,baseline.orientation,baseline.row,baseline.col]))
  return move
