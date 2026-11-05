"""Portable all-legal terminal-PPO inference; no training history required."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .engine import State,legal_moves
from .ai import features
from .value_search import encode
EXTRA=27
DEFAULT_GENERATION="gen2"
GENERATIONS=("gen2","2048")
def _candidate_inputs(s,qs,x=None,st=None):
    if x is None:x,st=features(s,qs)
    extra=np.zeros((len(qs),EXTRA),np.float32)
    for i,q in enumerate(qs):
        cells=q.move.cells;rs=[r for r,c in cells];cs=[c for r,c in cells]
        extra[i,q.move.piece]=1
        extra[i,21:]=[sum(rs)/len(rs)/13,sum(cs)/len(cs)/13,min(rs)/13,min(cs)/13,(max(rs)-min(rs))/4,(max(cs)-min(cs))/4]
    state=np.r_[st,np.array([s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],np.float32)]
    return np.concatenate((x,extra),axis=1),state.astype(np.float32)

def observe(s,qs=None):
 if s.done:raise ValueError('terminal has no policy actions')
 qs=legal_moves(s) if qs is None else qs;b,st=encode(s)
 if qs:
  x,_=_candidate_inputs(s,qs);cells=np.full((len(qs),5),196,np.int64)
  for i,q in enumerate(qs):
   for j,(y,col) in enumerate(q.move.cells):cells[i,j]=14*y+col
 else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
 return (b,st,x,cells),qs

def collate(obs,device='cuda'):
 n=max(len(o[2]) for o in obs);b=len(obs);x=np.zeros((b,n,39),np.float32);cells=np.full((b,n,5),196,np.int64);mask=np.zeros((b,n),bool)
 for i,o in enumerate(obs):k=len(o[2]);x[i,:k]=o[2];cells[i,:k]=o[3];mask[i,:k]=True
 return tuple(torch.as_tensor(a,device=device) for a in (np.stack([o[0] for o in obs]),np.stack([o[1] for o in obs]),x,cells,mask))

class TerminalNet(nn.Module):
 def __init__(self):
  super().__init__();self.conv=nn.Sequential(nn.Conv2d(6,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU());self.global_net=nn.Sequential(nn.Flatten(),nn.Linear(8*196,32),nn.Tanh());self.position=nn.Embedding(197,16,padding_idx=196)
  self.policy=nn.Sequential(nn.Linear(32+49+39+16+8,64),nn.Tanh(),nn.Linear(64,1));self.value=nn.Sequential(nn.Linear(32+49,64),nn.Tanh(),nn.Linear(64,1),nn.Tanh())
  nn.init.zeros_(self.policy[-1].weight);nn.init.zeros_(self.policy[-1].bias);nn.init.zeros_(self.value[-2].weight);nn.init.zeros_(self.value[-2].bias)
  self.register_buffer('prior',torch.tensor([12,4,2,1.2,2,.3,1,.2,-.6,0,0,.2],dtype=torch.float32))
 def forward(self,board,state,x,cells,mask):
  f=self.conv(board.float());g=self.global_net(f);b,n=x.shape[:2];count=(cells!=196).sum(-1).clamp_min(1).unsqueeze(-1)
  pos=self.position(cells).sum(-2)/count
  flat=nn.functional.pad(f.flatten(2),(0,1));idx=cells.reshape(b,1,-1).expand(-1,8,-1);local=flat.gather(2,idx).reshape(b,8,n,5).sum(-1).transpose(1,2)/count
  state_g=torch.cat((g,state),-1);parts=torch.cat((state_g[:,None].expand(-1,n,-1),x,pos,local),-1)
  logits=x[...,:12]@self.prior+self.policy(parts).squeeze(-1);return logits.masked_fill(~mask,-float('inf')),self.value(state_g).squeeze(-1)


class PPOAgent:
 def __init__(self,model_seed=1401,device='auto',checkpoint=None,generation=DEFAULT_GENERATION):
  if generation not in GENERATIONS:raise ValueError('generation must be gen2 or 2048')
  self.generation=generation
  if model_seed not in (1401,1402,1403):raise ValueError('model seed must be 1401,1402,1403')
  self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
  path=Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[1]/f'artifacts/ppo-{generation}/seed-{model_seed}.pt'
  ck=torch.load(path,map_location='cpu',weights_only=True)
  if ck.get('format')!='terminal-ppo-inference-v1':raise ValueError('not an exported PPO inference checkpoint')
  self.model=TerminalNet().to(self.device);self.model.load_state_dict(ck['model']);self.model.eval();self.metadata={k:v for k,v in ck.items() if k!='model'};self.times=[]
  with torch.no_grad():self.model(*collate([observe(State())[0]],self.device))
  if self.device.startswith('cuda'):torch.cuda.synchronize()
 def act(self,s,rng):
  start=time.perf_counter();obs,qs=observe(s)
  with torch.no_grad():logits,_=self.model(*collate([obs],self.device));scores=logits[0].cpu().numpy()
  ix=int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)));self.times.append(time.perf_counter()-start)
  return qs[ix].move if qs else None
