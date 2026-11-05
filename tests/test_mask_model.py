import json
import numpy as np
import torch
from blokus.mask_model import CandidateNet
from blokus.correction import corrected_values,masked_loss
from blokus.engine import State,Move,legal_moves
from blokus.root_policy import inputs
from blokus.value_search import encode

def fixture():
 r=json.load(open('artifacts/candidate-mask/collision.json'));s=State(**r['state']);by={q.move:q for q in legal_moves(s)};qs=[by[Move(*m)] for m in r['moves']];x,_=inputs(s,qs);cells=np.zeros((2,196),np.float32)
 for i,q in enumerate(qs):
  for y,col in q.move.cells:cells[i,y*14+col]=1
 b,st=encode(s)
 return torch.tensor(b)[None],torch.tensor(st)[None],torch.tensor(np.concatenate((x,cells),1))[None]

def test_real_summary_collision_and_spatial_gradient():
 b,st,x=fixture();assert torch.equal(x[0,0,:39],x[0,1,:39]);assert not torch.equal(x[0,0,39:],x[0,1,39:])
 torch.manual_seed(19);model=CandidateNet('cells');torch.nn.init.normal_(model.head[-1].weight)
 x.requires_grad_();v=model(b,st,x);assert not torch.equal(v[:,0],v[:,1]);v.sum().backward()
 assert x.grad[...,39:].abs().sum()>0;assert model.board[0].weight.grad[:,6].abs().sum()>0
 torch.manual_seed(19);old=CandidateNet('summary');torch.nn.init.normal_(old.head[-1].weight);v=old(b,st,x.detach());torch.testing.assert_close(v[:,0],v[:,1],rtol=0,atol=1e-6)

def test_shared_initialization_and_capacity():
 torch.manual_seed(1100);a=CandidateNet('summary');torch.manual_seed(1100);b=CandidateNet('cells')
 for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k][:,:6] if k=='board.0.weight' else b.state_dict()[k])
 assert sum(p.numel() for p in b.parameters())-sum(p.numel() for p in a.parameters())==72
 board,st,x=fixture();assert torch.equal(a(board,st,x),b(board,st,x))

def test_mask_unknown_and_final_application():
 p=torch.tensor([[2.,9.]],requires_grad=True);loss=masked_loss(p,torch.tensor([[1.,float('nan')]]),torch.tensor([[True,False]]));loss.backward();assert p.grad[0,1]==0
 v=corrected_values([1,2],[4,-4],[True,True],.5);assert np.argmax(v)==0
 v=corrected_values([1,2],[4,-4],[True,True],.5,[True,False]);assert v[0]==1

def test_same_search_final_correction_connection():
 import random
 from blokus.mask_model import CandidateAgent
 a=CandidateAgent(None,10000,device='cpu');s=State(remaining=[3,3]);a.act(s,random.Random(1));initial=a.initial_values.copy();known=a.last_known.copy();target=int(np.flatnonzero(known)[-1])
 class Forced(torch.nn.Module):
  def forward(self,b,st,x):
   assert x.shape[-1]==235
   v=torch.zeros(x.shape[:2]);v[:,target]=10000;return v
 a.model=Forced();a.act(s,random.Random(1));assert a.final_values.argmax()==target
 a.alpha=0;a.act(s,random.Random(1));assert np.allclose(a.final_values[known],initial[known])
