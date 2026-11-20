"""Research-only value models; frozen policy stays in a separate PPOAgent."""
import copy
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from .engine import PLACEMENTS,legal_moves
from .value_inputs import encode

@lru_cache(maxsize=1)
def placement_table():
    # Four bins by placement centroid, not four independent placement counts.
    rows=[]
    for q in PLACEMENTS:
        cells=q.move.cells
        quad=int(sum(r for r,c in cells)/len(cells)>=7)*2+int(sum(c for r,c in cells)/len(cells)>=7)
        rows.append((q.move.piece,quad))
    return np.asarray(rows,dtype=np.int64)

def mobility(s,backend='native'):
    """For each player/piece: available, log count, four count fractions (252)."""
    out=np.zeros((2,21,6),np.float32)
    if s.done:return out.reshape(-1)
    for side,player in enumerate((s.turn,1-s.turn)):
        z=copy.deepcopy(s);z.turn=player
        if backend=='native':
            from .ppo_native import scan
            rows=placement_table()[scan(z,False)[0]]
        elif backend=='python':
            rows=[]
            for q in legal_moves(z):
                cells=q.move.cells;quad=int(sum(r for r,c in cells)/len(cells)>=7)*2+int(sum(c for r,c in cells)/len(cells)>=7)
                rows.append((q.move.piece,quad))
            rows=np.asarray(rows,dtype=np.int64).reshape(-1,2)
        else:raise ValueError('backend must be python or native')
        if len(rows):
            counts=np.bincount(rows[:,0],minlength=21)
            bins=np.bincount(rows[:,0]*4+rows[:,1],minlength=84).reshape(21,4)
            out[side,:,0]=counts>0
            out[side,:,1]=np.log1p(counts)/np.log(4097.)
            out[side,:,2:]=bins/np.maximum(counts[:,None],1)
    return out.reshape(-1)

class PlacementValue(nn.Module):
    def __init__(self,source,variant):
        super().__init__()
        if variant not in ('A','B','C'):raise ValueError('unknown variant')
        if source.channels!=4:raise ValueError('requires gen3 four-channel model')
        self.variant=variant
        self.conv=copy.deepcopy(source.conv)
        self.global_net=copy.deepcopy(source.global_net)
        self.value=copy.deepcopy(source.value)
        if variant=='A':
            self.mobility_head=nn.Linear(252,64,bias=False)
            nn.init.zeros_(self.mobility_head.weight)
        if variant=='B':
            self.film=nn.Linear(42,32)
            nn.init.zeros_(self.film.weight);nn.init.zeros_(self.film.bias)
    def forward(self,board,state,extra=None):
        if self.variant=='B':
            mod=self.film(state[:,:42]).reshape(-1,2,2,8)
            f=self.conv[0](board.float())
            f=torch.relu(f*(1+mod[:,0,0,:,None,None])+mod[:,0,1,:,None,None])
            f=self.conv[2](f)
            f=torch.relu(f*(1+mod[:,1,0,:,None,None])+mod[:,1,1,:,None,None])
        else:f=self.conv(board.float())
        x=torch.cat((self.global_net(f),state),-1)
        h=self.value[0](x)
        if self.variant=='A':
            if extra is None:raise ValueError('A requires mobility features')
            h=h+self.mobility_head(extra)
        return self.value[3](self.value[2](self.value[1](h))).squeeze(-1)

class PlacementEvaluator:
    def __init__(self,model,backend='native'):
        self.model=model.eval();self.backend=backend
        self.device=next(model.parameters()).device
    def __call__(self,state):
        if state.done:
            d=state.score(state.turn)-state.score(1-state.turn)
            return float((d>0)-(d<0))
        b,st=encode(state)
        extra=mobility(state,self.backend) if self.model.variant=='A' else np.zeros(252,np.float32)
        with torch.inference_mode():
            return self.model(torch.as_tensor(b[None,:4],device=self.device),torch.as_tensor(st[None],device=self.device),torch.as_tensor(extra[None],device=self.device)).item()
