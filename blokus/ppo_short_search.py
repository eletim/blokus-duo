"""Selective batched two-ply WDL search. Completed batches only; no learning."""
import time
import numpy as np
import torch
from .terminal_ppo import TerminalAgent,observe,collate
from .value_search import encode
from .engine import play
from .research import clone


def terminal_value(s,player):
 return float(np.sign(s.score(player)-s.score(1-player)))

def known_terminal(s):return s.done or not any(s.remaining)


def choose_complete(values,margin):
 """All entries are completed same-depth root WDLs; index zero is policy fallback."""
 if not values or any(v is None or not np.isfinite(v) for v in values):return 0
 best=int(np.argmax(values))
 return best if values[best]>values[0]+margin else 0

class ShortSearch(TerminalAgent):
 def __init__(self,path,budget_ms=25,margin=0.,device='cuda'):
  super().__init__(path,device);self.budget_ms=budget_ms;self.width,self.reply=(4,2) if budget_ms==6 else (8,3);self.margin=margin;self.records=[];self.reset_game()
 def reset_game(self):
  self.cost={'prepare':.0007,'policy':.0008,'leaves':.00085};self.records=[]
 def sync(self):
  if str(self.device).startswith('cuda'):torch.cuda.synchronize()
 def policies(self,observations):
  with torch.no_grad():out,_=self.model(*collate(observations,self.device))
  self.sync();return [x[:len(o[2])].cpu().numpy() for x,o in zip(out,observations)]
 def values(self,states,player):
  out=[None]*len(states);indices=[i for i,s in enumerate(states) if not known_terminal(s)]
  if indices:
   obs=[encode(states[i]) for i in indices];board=torch.as_tensor(np.stack([x[0] for x in obs]),device=self.device);st=torch.as_tensor(np.stack([x[1] for x in obs]),device=self.device)
   with torch.no_grad():g=self.model.global_net(self.model.conv(board.float()));v=self.model.value(torch.cat([g,st],-1)).flatten()
   self.sync()
   for i,x in zip(indices,v.cpu().tolist()):out[i]=x if states[i].turn==player else -x
  for i,s in enumerate(states):
   if known_terminal(s):out[i]=terminal_value(s,player)
  return out
 def act(self,s,rng):
  start=time.perf_counter();deadline=start+self.budget_ms/1000;reserve=max(.0004,self.budget_ms*.00008)
  rec=dict(completed=False,fallback=False,forced=False,roots=0,replies=0,switched=False,reason='',root_ms=0.,prepare_ms=0.,policy_ms=0.,leaf_ms=0.)
  obs,qs=observe(s);root_prepare=time.perf_counter()-start;rec['root_prepare_ms']=root_prepare*1000;logits=self.policies([obs])[0];rec['root_ms']=(time.perf_counter()-start)*1000
  if not qs:order=[];fallback=None;rec['forced']=True
  else:
   # Preserve network-only random tie policy, then stable policy ordering.
   ix=rng.choice(np.flatnonzero(logits>=logits.max()-1e-7));order=[int(ix)]+[int(i) for i in np.argsort(-logits,kind='stable') if i!=ix];order=order[:self.width];fallback=qs[ix].move;rec['forced']=len(qs)==1
  result=fallback
  def fits(cost):return time.perf_counter()+cost+reserve<deadline
  if len(order)>1:
   self.cost['prepare']=max(.0001,min(self.cost['prepare'],root_prepare*1.25))
   estimate=len(order)*self.cost['prepare']+self.cost['policy']+self.cost['leaves']
   if not fits(estimate):rec['reason']='preflight'
   else:
    children=[];observations=[];replies=[];t=time.perf_counter()
    for ix in order:
     if not fits(self.cost['prepare']+self.cost['policy']+self.cost['leaves']):break
     child=clone(s);play(child,qs[ix].move,validate=False);children.append(child);o,r=observe(child);observations.append(o);replies.append(r)
    rec['prepare_ms']=(time.perf_counter()-t)*1000
    if children:self.cost['prepare']=max(.0001,(time.perf_counter()-t)/len(children)*1.2)
    if len(children)!=len(order) or not fits(self.cost['policy']+self.cost['leaves']):rec['reason']='prepare'
    else:
     t=time.perf_counter();policies=self.policies(observations);elapsed=time.perf_counter()-t;rec['policy_ms']=elapsed*1000;self.cost['policy']=max(.0002,elapsed*1.2)
     leaves=[];groups=[]
     for child,r,logit in zip(children,replies,policies):
      group=[]
      if known_terminal(child):group=[len(leaves)];leaves.append(child)
      else:
       for j in np.argsort(-logit,kind='stable')[:self.reply] if r else [None]:
        leaf=clone(child);play(leaf,r[j].move if j is not None else None,validate=False);group.append(len(leaves));leaves.append(leaf)
      groups.append(group)
     if not fits(self.cost['leaves']):rec['reason']='leaves'
     else:
      t=time.perf_counter();v=self.values(leaves,s.turn);elapsed=time.perf_counter()-t;rec['leaf_ms']=elapsed*1000;self.cost['leaves']=max(.0002,elapsed*1.2)
      vals=[min(v[j] for j in group) for group in groups]
      if time.perf_counter()>=deadline:rec['reason']='late_complete_batch_discarded'
      else:
       best=choose_complete(vals,self.margin);result=qs[order[best]].move;rec.update(completed=True,roots=len(order),replies=len(leaves),switched=best!=0,values=vals)
  rec['fallback']=len(order)>1 and not rec['completed'];rec['elapsed_ms']=(time.perf_counter()-start)*1000;rec['overrun']=rec['elapsed_ms']>self.budget_ms;self.records.append(rec);self.times.append(rec['elapsed_ms']/1000);return result
