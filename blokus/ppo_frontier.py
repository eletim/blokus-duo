"""Suspendable reference DFS. Each generator owns all bounds and caches.
No speculative siblings, IPC, or scheduling-dependent RNG.
"""
import time,copy
import torch
import numpy as np
from .ppo_value_leaf import value_inputs
from .engine import legal_moves,play
from .ppo_policy import collate
from .ppo_value_leaf import value_only
def key(s):return (tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)

def child(s,m):
 c=copy.deepcopy(s);play(c,m,validate=False);return c

class FrontierSearch:
 def __init__(self,policy,budget_ms=500,width=4,max_depth=6,prune=True,value_evaluator=None):
  if budget_ms<=0 or width<1 or max_depth not in (2,4,6):raise ValueError('invalid limits')
  self.value_evaluator=value_evaluator;self.policy=policy;self.budget_ms=budget_ms;self.width=width;self.max_depth=max_depth;self.prune=prune;self.stats={}
 def act(self,s,rng):
  start=time.perf_counter();deadline=start+self.budget_ms/1000;tt={};pol={};leaves={};stats=dict(depth=0,nodes=0,nn_policy_calls=0,nn_value_calls=0,cache_hits=0,cutoffs=0,stages=[],fallback=True,forced_pass=False)
  def check():
   if time.perf_counter()>=deadline:raise TimeoutError
  def evaluate(z,leaf):
   check();stats['nodes']+=1;k=key(z)
   if z.done:return [],value_only(self.policy,z)
   if leaf:
    if k in leaves:stats['cache_hits']+=1;return [],leaves[k]
    stats['nn_value_calls']+=1;v=yield ('value',value_inputs(z,self.policy.channels));check();leaves[k]=v;return [],v
   if k in pol:stats['cache_hits']+=1;return pol[k]
   obs,qs=self.policy._observation(z)
   logits,v=yield ('policy',obs)
   order=torch.as_tensor(logits).argsort(descending=True).tolist()
   stats['nn_policy_calls']+=1;check();out=([qs[i].move for i in order[:self.width]] if qs else [None],v);pol[k]=out
   if self.value_evaluator is None:leaves[k]=v
   return out
  def visit(z,d,alpha,beta):
   check();k=(key(z),d,self.width);a0,b0=alpha,beta
   if self.prune and k in tt:
    value,flag=tt[k];stats['cache_hits']+=1
    if flag=='EXACT':return value
    # Use bounds only for a proven cutoff; do not narrow the window.
    # Classification below therefore stays relative to the searched window.
    if flag=='LOWER' and value>=beta:return value
    if flag=='UPPER' and value<=alpha:return value
   moves,v=yield from evaluate(z,not d)
   if d and not z.done:
    v=-float('inf')
    for m in moves:
     q=-(yield from visit(child(z,m),d-1,-beta,-alpha)) if self.prune else -(yield from visit(child(z,m),d-1,-float('inf'),float('inf')))
     v=max(v,q);alpha=max(alpha,v)
     if self.prune and alpha>=beta:stats['cutoffs']+=1;break
   if self.prune:tt[k]=(v,'UPPER' if v<=a0 else 'LOWER' if v>=b0 else 'EXACT')
   return v
  obs,qs=self.policy._observation(s)
  logits,_=yield ('policy',obs)
  ix=int(rng.choice(np.flatnonzero(logits>=logits.max()-1e-7)))
  selected=qs[ix].move if qs else None
  if not legal_moves(s):stats['forced_pass']=True
  else:
   try:
    roots,_=yield from evaluate(s,False)
    if selected not in roots:roots[-1]=selected
    for d in range(2,self.max_depth+1,2):
     scores=[]
     for m in roots:scores.append(-(yield from visit(child(s,m),d-1,-float('inf'),float('inf'))))
     check()
     selected=roots[max(range(len(roots)),key=scores.__getitem__)];stats.update(depth=d,root_scores=scores,fallback=False);stats['stages'].append(d)
   except TimeoutError:pass
  stats['elapsed_ms']=(time.perf_counter()-start)*1000;self.stats=stats;return selected

class FrontierEvaluator:
 """Evaluate ready requests from one frozen model; never mixes model seeds."""
 def __init__(self,policy):
  self.policy=policy;self.hist={};self.pack_seconds=0.;self.forward_seconds=0.;self.return_seconds=0.
 def batch(self,requests):
  p=self.policy;answers=[None]*len(requests)
  for kind in ('policy','value'):
   ix=[i for i,r in enumerate(requests) if r[0]==kind]
   if not ix:continue
   n=len(ix);self.hist[n]=self.hist.get(n,0)+1;t=time.perf_counter()
   if kind=='policy':inputs=collate([requests[i][1] for i in ix],p.device)
   else:
    inputs=[torch.as_tensor(np.stack([requests[i][1][j] for i in ix]),device=p.device) for j in (0,1)]
   if str(p.device).startswith('cuda'):torch.cuda.synchronize()
   self.pack_seconds+=time.perf_counter()-t;t=time.perf_counter()
   with torch.inference_mode():
    if kind=='policy':l,v=p.model(*inputs)
    else:
     b,st=inputs;g=p.model.global_net(p.model.conv(b.float()));v=p.model.value(torch.cat((g,st),-1)).squeeze(-1)
   if str(p.device).startswith('cuda'):torch.cuda.synchronize()
   self.forward_seconds+=time.perf_counter()-t;t=time.perf_counter()
   vv=v.cpu().numpy();ll=l.cpu().numpy() if kind=='policy' else None
   for j,i in enumerate(ix):answers[i]=(ll[j,:len(requests[i][1][2])].copy(),float(vv[j])) if kind=='policy' else float(vv[j])
   self.return_seconds+=time.perf_counter()-t
  return answers

def run_frontier(generators,evaluator):
 """Drive arbitrarily many independent search/game coroutines in one process."""
 active=[];results=[None]*len(generators)
 for i,g in enumerate(generators):
  try:active.append((i,g,next(g)))
  except StopIteration as end:results[i]=end.value
 while active:
  replies=evaluator.batch([x[2] for x in active]);pending=[]
  for (i,g,_),answer in zip(active,replies):
   try:pending.append((i,g,g.send(answer)))
   except StopIteration as end:results[i]=end.value
  active=pending
 return results
