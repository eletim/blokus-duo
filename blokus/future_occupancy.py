"""Research terminal occupancy prediction; current-state inputs only."""
import copy
import numpy as np
import torch
from torch import nn
from .value_inputs import encode
from .placement_value import mobility
from .engine import cells_of

class FutureValue(nn.Module):
 def __init__(self,source,arm):
  super().__init__();assert arm in 'ABC';self.arm=arm
  for name in ['conv','global_net','value','mobility_head']:setattr(self,name,copy.deepcopy(getattr(source,name)))
  self.occupancy=nn.Conv2d(8,3,1)
  self.occupancy_state=nn.Linear(49,3*196)
  self.map_value=nn.Linear(3*196,64,bias=False);nn.init.zeros_(self.map_value.weight)
 def forward(self,board,state,extra,with_map=True):
  f=self.conv(board.float());x=torch.cat((self.global_net(f),state),-1)
  h=self.value[0](x)+self.mobility_head(extra);logits=None;probs=None
  if with_map or self.arm=='C':
   logits=self.occupancy(f)+self.occupancy_state(state).reshape(-1,3,14,14)
   probs=logits.softmax(1);occupied=board[:,:2].sum(1,keepdim=True).bool()
   known=torch.cat((board[:,:2],torch.zeros_like(board[:,:1])),1).float()
   probs=torch.where(occupied,known,probs)
  if self.arm=='C':h=h+self.map_value(probs.flatten(1))
  v=self.value[3](self.value[2](self.value[1](h))).squeeze(-1)
  return v,logits,probs

def terminal_labels(final_boards,turn):
 y=np.full((14,14),2,np.int64)
 for cls,p in enumerate([turn,1-turn]):
  for r,c in cells_of(final_boards[p]):y[r,c]=cls
 return y

def transform(x,k):
 """D4 on final two spatial dimensions, shared by inputs and labels."""
 z=np.rot90(x,k%4,axes=(-2,-1))
 return np.flip(z,axis=-1).copy() if k>=4 else z.copy()

class FutureEvaluator:
 def __init__(self,model):self.model=model.cpu().eval()
 def __call__(self,s):
  if s.done:
   d=s.score(s.turn)-s.score(1-s.turn);return float((d>0)-(d<0))
  b,st=encode(s);ex=mobility(s,'native')
  with torch.inference_mode():return self.model(torch.from_numpy(b[None,:4]),torch.from_numpy(st[None]),torch.from_numpy(ex[None]),with_map=False)[0].item()
