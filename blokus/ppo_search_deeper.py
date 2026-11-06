"""Fixed policy-width tree, complete iterative depths, bounded TT, terminal WDL."""
import time,copy
import torch
from .engine import legal_moves,play
from .ppo_policy import collate
from .ppo_value_leaf import value_only
from .ppo_search_staged import key

def child(s,m):
 c=copy.deepcopy(s);play(c,m,validate=False);return c

class DeepSearch:
 def __init__(self,policy,budget_ms=500,width=4,max_depth=6,prune=True):
  if budget_ms<=0 or width<1 or max_depth not in (2,4,6):raise ValueError('invalid limits')
  self.policy=policy;self.budget_ms=budget_ms;self.width=width;self.max_depth=max_depth;self.prune=prune;self.stats={}
 def act(self,s,rng):
  start=time.perf_counter();deadline=start+self.budget_ms/1000;tt={};pol={};leaves={};stats=dict(depth=0,nodes=0,nn_policy_calls=0,nn_value_calls=0,cache_hits=0,cutoffs=0,stages=[],fallback=True,forced_pass=False)
  def check():
   if time.perf_counter()>=deadline:raise TimeoutError
  def evaluate(z,leaf):
   check();stats['nodes']+=1;k=key(z)
   if z.done:return [],value_only(self.policy,z)
   if leaf:
    if k in leaves:stats['cache_hits']+=1;return [],leaves[k]
    stats['nn_value_calls']+=1;v=value_only(self.policy,z);check();leaves[k]=v;return [],v
   if k in pol:stats['cache_hits']+=1;return pol[k]
   obs,qs=self.policy._observation(z)
   with torch.inference_mode():
    l,v=self.policy.model(*collate([obs],self.policy.device));order=l[0].argsort(descending=True).cpu().tolist();v=v.item()
   stats['nn_policy_calls']+=1;check();out=([qs[i].move for i in order[:self.width]] if qs else [None],v);pol[k]=out;leaves[k]=v;return out
  def visit(z,d,alpha,beta):
   check();k=(key(z),d,self.width);a0,b0=alpha,beta
   if self.prune and k in tt:
    value,flag=tt[k];stats['cache_hits']+=1
    if flag=='EXACT':return value
    # Use bounds only for a proven cutoff; do not narrow the window.
    # Classification below therefore stays relative to the searched window.
    if flag=='LOWER' and value>=beta:return value
    if flag=='UPPER' and value<=alpha:return value
   moves,v=evaluate(z,not d)
   if d and not z.done:
    v=-float('inf')
    for m in moves:
     q=-visit(child(z,m),d-1,-beta,-alpha) if self.prune else -visit(child(z,m),d-1,-float('inf'),float('inf'))
     v=max(v,q);alpha=max(alpha,v)
     if self.prune and alpha>=beta:stats['cutoffs']+=1;break
   if self.prune:tt[k]=(v,'UPPER' if v<=a0 else 'LOWER' if v>=b0 else 'EXACT')
   return v
  selected=self.policy.act(s,rng)
  if not legal_moves(s):stats['forced_pass']=True
  else:
   try:
    roots,_=evaluate(s,False)
    if selected not in roots:roots[-1]=selected
    for d in range(2,self.max_depth+1,2):
     scores=[-visit(child(s,m),d-1,-float('inf'),float('inf')) for m in roots];check()
     selected=roots[max(range(len(roots)),key=scores.__getitem__)];stats.update(depth=d,root_scores=scores,fallback=False);stats['stages'].append(d)
   except TimeoutError:pass
  stats['elapsed_ms']=(time.perf_counter()-start)*1000;self.stats=stats;return selected
