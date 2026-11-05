"""Root-point residual after completed 2ply values; fixed reply/value policy."""
import time
import numpy as np
import torch
from torch import nn
from .root_policy import inputs,StudentAgent,XDIM
from .value_search import encode,exact_terminal
from .engine import legal_moves,play
from .research import clone
from .ai import features,heuristic

class CorrectionNet(nn.Module):
 def __init__(self):
  super().__init__()
  self.board=nn.Sequential(nn.Conv2d(6,8,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(8*196,32),nn.Tanh())
  self.head=nn.Sequential(nn.Linear(32+49+XDIM,64),nn.Tanh(),nn.Linear(64,1))
  nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
 def forward(self,board,state,x):
  z=torch.cat((self.board(board.float()),state),-1)[:,None,:].expand(-1,x.shape[1],-1)
  return self.head(torch.cat((z,x),-1)).squeeze(-1)

def masked_loss(pred,target,mask):
 if not bool(mask.any()):raise ValueError('empty labels')
 # Index BEFORE subtraction: masked NaNs must not enter forward or backward.
 return nn.functional.smooth_l1_loss(pred[mask],target[mask])

def corrected_values(initial,delta,known,alpha=1.,terminal=None):
 v=np.asarray(initial,float).copy();known=np.asarray(known,bool)
 ok=known & np.isfinite(v) & np.isfinite(delta)
 if terminal is not None:ok &= ~np.asarray(terminal,bool)
 v[ok]+=alpha*np.asarray(delta)[ok]
 v[~known]=-np.inf
 return v

class CorrectionAgent(StudentAgent):
 def __init__(self,checkpoint,budget_ms=25,policy_path=None,device='cuda',alpha=1.):
  super().__init__(checkpoint,budget_ms,None,device);self.alpha=alpha;self.model=CorrectionNet().to(device)
  if policy_path:self.model.load_state_dict(torch.load(policy_path,map_location='cpu',weights_only=False)['model'])
  self.model.eval()
  with torch.no_grad():self.model(torch.zeros(1,6,14,14,device=device),torch.zeros(1,49,device=device),torch.zeros(1,128,XDIM,device=device))
  if device.startswith('cuda'):torch.cuda.synchronize()
 def act(self,s,rng):
  if s.done:raise ValueError('terminal')
  start=time.perf_counter();deadline=start+self.budget_ms/1000;qs=legal_moves(s);generation=time.perf_counter()-start
  if not qs:
   self.times.append(time.perf_counter()-start);self.decisions.append(dict(ms=self.times[-1]*1000,policy_ms=0.,completed_roots=0,candidates=0,fallback=False));return None
  rank_start=time.perf_counter();x,st=features(s,qs);members=np.argsort(-heuristic(x),kind='stable')[:128]
  t=time.perf_counter();xx,_=inputs(s,[qs[int(i)] for i in members],x[members],st);board,state=encode(s)
  with torch.no_grad():delta=self.model(torch.as_tensor(board[None],device=self.device),torch.as_tensor(state[None],device=self.device),torch.as_tensor(xx[None],device=self.device))[0].cpu().numpy()
  policy_ms=(time.perf_counter()-t)*1000;self.sections['legal']+=generation;self.sections['rank']+=time.perf_counter()-rank_start
  values=np.full(len(members),np.nan);known=np.zeros(len(members),bool);terminal=np.zeros(len(members),bool)
  for ordinal,i in enumerate(members):
   if time.perf_counter()+self.cost_estimate*1.2+.0001>=deadline:break
   step=time.perf_counter();c=clone(s);play(c,qs[int(i)].move,validate=False);self.nodes+=1;terminal[ordinal]=exact_terminal(c)
   replies=legal_moves(c);leaves=[]
   if replies:
    rx,_=features(c,replies)
    for j in np.argsort(-heuristic(rx),kind='stable')[:3]:
     leaf=clone(c);play(leaf,replies[int(j)].move,validate=False);leaves.append(leaf);self.nodes+=1
   else:
    leaf=clone(c);play(leaf,None,validate=False);leaves=[leaf];self.nodes+=1
   t=time.perf_counter();self.sections['expand']+=t-step;v=float(self.evaluator.batch(leaves).min())*40;self.sections['value']+=time.perf_counter()-t
   cost=time.perf_counter()-step;cap=self.budget_ms/1000;self.cost_estimate=max(.00015,min(cap*.45,.75*self.cost_estimate+.25*min(cost,cap)))
   if time.perf_counter()>deadline:break
   values[ordinal]=v;known[ordinal]=True;self.roots+=1
  final=corrected_values(values,delta,known,self.alpha,terminal)
  if known.any():choice=int(rng.choice(np.flatnonzero(final>=np.max(final)-4e-6)))
  else:choice=0;self.fallbacks+=1
  self.initial_values=values;self.final_values=final;self.last_delta=delta;self.last_known=known
  self.times.append(time.perf_counter()-start);self.decisions.append(dict(ms=self.times[-1]*1000,policy_ms=policy_ms,generation_ms=generation*1000,completed_roots=int(known.sum()),candidates=len(members),fallback=not bool(known.any())))
  return qs[int(members[choice])].move
