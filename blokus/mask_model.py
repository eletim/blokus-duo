"""Spatial action conditioning. Last 196 x entries encode occupied board cells."""
import torch
from torch import nn
from .correction import CorrectionNet

class CandidateNet(CorrectionNet):
 def __init__(self,representation='summary'):
  super().__init__();self.representation=representation
  if representation=='cells':
   # Preserve every common initial weight; only the new channel is sampled.
   old=self.board[0];new=nn.Conv2d(7,8,3,padding=1)
   with torch.no_grad():new.weight[:,:6].copy_(old.weight);new.bias.copy_(old.bias)
   self.board[0]=new
 def forward(self,board,state,x):
  if self.representation=='summary':return super().forward(board,state,x[...,:39])
  b,n=x.shape[:2];cells=x[...,39:].reshape(b,n,1,14,14)
  spatial=torch.cat((board.float()[:,None].expand(-1,n,-1,-1,-1),cells),2).reshape(b*n,7,14,14)
  z=self.board(spatial).reshape(b,n,32)
  return self.head(torch.cat((z,state[:,None].expand(-1,n,-1),x[...,:39]),-1)).squeeze(-1)

from .correction import CorrectionAgent
from .root_policy import inputs
import numpy as np

def spatial_inputs(s,qs,x=None,st=None):
 summary,_=inputs(s,qs,x,st);cells=np.zeros((len(qs),196),np.float32)
 for i,q in enumerate(qs):
  for y,col in q.move.cells:cells[i,y*14+col]=1
 return np.concatenate((summary,cells),1)

class CandidateAgent(CorrectionAgent):
 """Identical final correction/search path; representation changes only input/model."""
 def __init__(self,checkpoint,budget_ms=25,policy_path=None,device='cuda',alpha=.5,representation='cells'):
  super().__init__(checkpoint,budget_ms,None,device,alpha)
  self.model=CandidateNet(representation).to(device)
  if policy_path:self.model.load_state_dict(torch.load(policy_path,map_location='cpu',weights_only=False)['model'])
  self.model.eval()
  with torch.no_grad():self.model(torch.zeros(1,6,14,14,device=device),torch.zeros(1,49,device=device),torch.zeros(1,128,235,device=device))
  if device.startswith('cuda'):torch.cuda.synchronize()
 def candidate_inputs(self,s,qs,x,st):return spatial_inputs(s,qs,x,st)
