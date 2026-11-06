"""Experimental update-only compact candidates; unchanged weights and dense distribution.

Only real candidates enter embedding/gather/MLP. Scatter logits back to the old
per-state shape to preserve Categorical, per-state entropy/loss, and action order.
Collection/sampling remain on the reference path. Floating reductions can differ.
"""
import numpy as np
import torch
from torch import nn

class CompactBatch:
 def __init__(self,obs,device):
  lengths=[len(o[2]) for o in obs];self.shape=(len(obs),max(lengths))
  self.board=torch.as_tensor(np.stack([o[0] for o in obs]),device=device)
  self.state=torch.as_tensor(np.stack([o[1] for o in obs]),device=device)
  self.x=torch.as_tensor(np.concatenate([o[2] for o in obs]),device=device)
  self.cells=torch.as_tensor(np.concatenate([o[3] for o in obs]),device=device)
  owner=np.repeat(np.arange(len(obs)),lengths)
  self.owner=torch.as_tensor(owner,device=device)
  self.destination=torch.as_tensor(np.concatenate([i*self.shape[1]+np.arange(n) for i,n in enumerate(lengths)]),device=device)
  self.cell_index=self.owner[:,None]*197+self.cells

def forward(model,batch):
 f=model.conv(batch.board.float());g=model.global_net(f)
 state_g=torch.cat((g,batch.state),-1)
 count=(batch.cells!=196).sum(-1).clamp_min(1).unsqueeze(-1)
 pos=model.position(batch.cells).sum(-2)/count
 table=nn.functional.pad(f.flatten(2),(0,1)).transpose(1,2).reshape(-1,8)
 local=table.index_select(0,batch.cell_index.flatten()).reshape(-1,5,8).sum(1)/count
 parts=torch.cat((state_g.index_select(0,batch.owner),batch.x,pos,local),-1)
 scores=batch.x[:,:12]@model.prior+model.policy(parts).squeeze(-1)
 logits=scores.new_full((batch.shape[0]*batch.shape[1],),-float('inf')).scatter(0,batch.destination,scores).reshape(batch.shape)
 return logits,model.value(state_g).squeeze(-1)
