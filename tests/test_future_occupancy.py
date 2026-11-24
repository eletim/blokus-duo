import copy
import numpy as np
import torch
from blokus.engine import State,play,legal_moves
from blokus.value_inputs import encode
from blokus.future_occupancy import FutureValue,terminal_labels,transform
from blokus.gen4 import load
from blokus.placement_value import mobility

def test_perspective_symmetry_and_occupied():
 s=State()
 for _ in range(10):play(s,legal_moves(s)[0].move,validate=True)
 y=terminal_labels(s.boards,s.turn);z=terminal_labels(s.boards,1-s.turn)
 assert np.array_equal(z,np.where(y==2,2,1-y))
 b,st=encode(s)
 for k in range(8):
  t=transform(y,k);a=transform(b,k);assert np.all(t[a[0].astype(bool)]==0);assert np.all(t[a[1].astype(bool)]==1)
 torch.set_num_threads(1);_,v=load(1751401,'native');model=FutureValue(v.model,'C')
 with torch.no_grad():
  _,_,p=model(torch.tensor(b[None,:4]),torch.tensor(st[None]),torch.tensor(mobility(s)[None]))
 assert torch.equal(p[0,0][torch.tensor(b[0],dtype=torch.bool)],torch.ones(int(b[0].sum())))
 assert torch.allclose(p.sum(1),torch.ones((1,14,14)))

def test_arm_isolation_and_fresh_value_identity():
 torch.set_num_threads(1);s=State();_,v=load(1751401,'native');b,st=encode(s);args=(torch.tensor(b[None,:4]),torch.tensor(st[None]),torch.tensor(mobility(s)[None]))
 for arm in 'ABC':
  torch.manual_seed(7);m=FutureValue(v.model,arm);before=m(*args)[0]
  assert torch.equal(before,v.model(*args))
  with torch.no_grad():m.occupancy.bias.add_(torch.tensor([4.,-4.,1.]))
  assert torch.equal(before,m(*args)[0]) # C zero projection starts identical too
  if arm=='C':
   with torch.no_grad():m.map_value.weight.normal_(0,.01)
   a=m(*args)[0]
   with torch.no_grad():m.occupancy.bias.mul_(-1)
   assert not torch.equal(a,m(*args)[0])
