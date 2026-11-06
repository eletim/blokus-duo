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
from .placement_geometry import geometry
from .value_search import encode
EXTRA=27
DEFAULT_GENERATION="gen2"
GENERATIONS=("gen2","2048")
def _candidate_inputs(s,qs,x=None,st=None):
    if x is None:x,st=features(s,qs)
    extra=np.asarray([geometry(q.move)[2] for q in qs],dtype=np.float32).reshape(-1,EXTRA)
    state=np.r_[st,np.array([s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],np.float32)]
    return np.concatenate((x,extra),axis=1),state.astype(np.float32)

def observe(s,qs=None,channels=6):
 if s.done:raise ValueError('terminal has no policy actions')
 if channels not in (4,6):raise ValueError('channels must be 4 or 6')
 qs=legal_moves(s) if qs is None else qs;b,st=encode(s);b=b[:channels].copy()
 if qs:
  x,_=_candidate_inputs(s,qs);cells=np.asarray([geometry(q.move)[3] for q in qs],dtype=np.int64)
 else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
 return (b,st,x,cells),qs

def collate(obs,device='cuda'):
 n=max(len(o[2]) for o in obs);b=len(obs);x=np.zeros((b,n,39),np.float32);cells=np.full((b,n,5),196,np.int64);mask=np.zeros((b,n),bool)
 for i,o in enumerate(obs):k=len(o[2]);x[i,:k]=o[2];cells[i,:k]=o[3];mask[i,:k]=True
 return tuple(torch.as_tensor(a,device=device) for a in (np.stack([o[0] for o in obs]),np.stack([o[1] for o in obs]),x,cells,mask))

class TerminalNet(nn.Module):
 def __init__(self,channels=6):
  if channels not in (4,6):raise ValueError('channels must be 4 or 6')
  super().__init__();self.channels=channels;self.conv=nn.Sequential(nn.Conv2d(channels,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU());self.global_net=nn.Sequential(nn.Flatten(),nn.Linear(8*196,32),nn.Tanh());self.position=nn.Embedding(197,16,padding_idx=196)
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
 def __init__(self,model_seed=1401,device='auto',checkpoint=None,generation=DEFAULT_GENERATION,backend="python"):
  if generation not in GENERATIONS:raise ValueError('generation must be gen2 or 2048')
  if backend not in ("python","native","auto"):raise ValueError("unknown backend")
  self.backend=backend;self.generation=generation
  if model_seed not in (1401,1402,1403):raise ValueError('model seed must be 1401,1402,1403')
  self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device
  path=Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[1]/f'artifacts/ppo-{generation}/seed-{model_seed}.pt'
  ck=torch.load(path,map_location='cpu',weights_only=True)
  fmt=ck.get('format');channels=6 if fmt=='terminal-ppo-inference-v1' else ck.get('channels') if fmt=='terminal-ppo-inference-v2' else None
  if channels not in (4,6) or ck.get('channels',channels)!=channels or ck['model']['conv.0.weight'].shape[1]!=channels:raise ValueError('checkpoint format/channel mismatch')
  self.channels=channels;self.model=TerminalNet(channels).to(self.device);self.model.load_state_dict(ck['model']);self.model.eval();self.metadata={k:v for k,v in ck.items() if k!='model'};self.times=[]
  with torch.no_grad():self.model(*collate([self._observation(State())[0]],self.device))
  if self.device.startswith('cuda'):torch.cuda.synchronize()
 def _observation(self,s):
  if self.backend=="python":return observe(s) if self.channels==6 else observe(s,channels=4)
  from .ppo_native import observe as native_observe
  return native_observe(s,self.channels,self.backend)
 def act(self,s,rng):
  start=time.perf_counter();obs,qs=self._observation(s)
  with torch.no_grad():logits,_=self.model(*collate([obs],self.device));scores=logits[0].cpu().numpy()
  ix=int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)));self.times.append(time.perf_counter()-start)
  return qs[ix].move if qs else None
