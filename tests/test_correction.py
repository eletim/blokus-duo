import numpy as np
import torch
import random
from blokus.correction import CorrectionNet,CorrectionAgent,corrected_values,masked_loss
from blokus.engine import State,legal_moves

def test_nan_mask_and_ties():
 p=torch.tensor([[1.,2.,3.]],requires_grad=True);y=torch.tensor([[1.,float('nan'),3.]])
 loss=masked_loss(p,y,torch.tensor([[True,False,True]]));loss.backward();assert loss==0 and p.grad[0,1]==0
 v=corrected_values([2,3,np.nan],[2,1,100],[1,1,0]);assert list(np.flatnonzero(v==v.max()))==[0,1]

def test_sign_terminal_unknown():
 assert np.allclose(corrected_values([10,-10],[3,-3],[1,1]),[13,-13])
 v=corrected_values([20,0],[100,100],[1,0],terminal=[1,0]);assert v[0]==20 and v[1]==-np.inf

def test_final_selection_not_overwritten():
 s=State(remaining=[3,3]);a=CorrectionAgent(None,10000,device='cpu');a.act(s,random.Random(1))
 initial=a.initial_values.copy();known=a.last_known.copy();target=int(np.flatnonzero(known)[-1])
 class Fixed(torch.nn.Module):
  def forward(self,b,st,x):
   out=torch.zeros(x.shape[:2]);out[:,target]=10000;return out
 a.model=Fixed();m=a.act(s,random.Random(1));assert a.final_values.argmax()==target
 from blokus.ai import features,heuristic
 qs=legal_moves(s);order=np.argsort(-heuristic(features(s,qs)[0]),kind='stable')[:128]
 assert m==qs[int(order[target])].move
 a.alpha=0;a.act(s,random.Random(1));assert np.allclose(a.final_values[known],initial[known])

def test_zero_init_and_pass():
 m=CorrectionNet();assert torch.count_nonzero(m(torch.rand(2,6,14,14),torch.rand(2,49),torch.rand(2,4,39)))==0
 a=CorrectionAgent(None,25,device='cpu');s=State(remaining=[0,1]);assert a.act(s,random.Random(1)) is None
