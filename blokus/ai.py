import random, time
import numpy as np
import torch
from torch import nn
from .engine import *
from .placement_geometry import geometry
# Candidate features are interpreted in current player's frame, not absolute colour.
DIM=12
STATE_DIM=46
TIMES={'generation':0.,'features':0.,'network':0.,'moves':0,'candidates':0}
def candidates(s):
    t=time.perf_counter(); qs=legal_moves(s); TIMES['generation']+=time.perf_counter()-t
    TIMES['moves']+=1; TIMES['candidates']+=len(qs); return qs

def features(s,qs):
    t=time.perf_counter(); p=s.turn; own=s.boards[p]; occ=own|s.boards[1-p]
    f,e=frontier(s,p); of,oe=frontier(s,1-p)
    rows=[];fc=f.bit_count();remaining=s.remaining[p].bit_count()/21
    difference=(s.boards[p].bit_count()-s.boards[1-p].bit_count())/89
    for q in qs:
        size=SIZES[q.move.piece];center,reach,_,_=geometry(q.move)
        nf=(f|q.corners)&~(occ|q.bits|e|q.edges)
        nfc=nf.bit_count();freed=nfc-fc
        rows.append([size/5,freed/10,(q.bits&of).bit_count()/5,center,reach[p],
                     (q.edges&of).bit_count()/10, nfc/40,
                     (q.corners&~occ&~q.bits).bit_count()/20,
                     float(size==1),remaining,
                     difference,
                     (q.edges&s.boards[1-p]).bit_count()/10])
    st=np.array([float(s.remaining[a]>>i&1) for a in (p,1-p) for i in range(21)]+[own.bit_count()/89,s.boards[1-p].bit_count()/89,f.bit_count()/40,of.bit_count()/40],dtype=np.float32)
    x=np.asarray(rows,dtype=np.float32).reshape(-1,DIM)
    TIMES['features']+=time.perf_counter()-t
    return x,st

def heuristic(x):
    # Large pieces, new own corners, blocking opponent frontier, expansion.
    return x @ np.array([12,4,2,1.2,2,0.3,1,0.2,-0.6,0,0,0.2],dtype=np.float32)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy=nn.Sequential(nn.Linear(DIM+STATE_DIM,96),nn.Tanh(),nn.Linear(96,64),nn.Tanh(),nn.Linear(64,1))
        self.value=nn.Sequential(nn.Linear(STATE_DIM,64),nn.Tanh(),nn.Linear(64,1),nn.Tanh())
    def forward(self,x,s):
        if s.ndim==1: s=s.unsqueeze(0)
        return self.policy(torch.cat((x,s.expand(x.shape[0],-1)),1)).squeeze(-1),self.value(s).squeeze()
def tensors(x,s,device): return torch.as_tensor(x,device=device),torch.as_tensor(s,device=device)
def select(s,kind,rng,model=None,device='cpu'):
    qs=candidates(s)
    if not qs:return None
    if kind=='random':return rng.choice(qs).move
    x,st=features(s,qs)
    if kind=='heuristic': scores=heuristic(x)
    else:
        t=time.perf_counter()
        with torch.no_grad(): scores=model(*tensors(x,st,device))[0].cpu().numpy()
        TIMES['network']+=time.perf_counter()-t
    # Randomized tie-breaking, reproducible with supplied RNG.
    best=np.flatnonzero(scores>=scores.max()-1e-7)
    return qs[int(rng.choice(best))].move

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
def save(path,model,opt,episode,phase):
    import os
    payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),episode=episode,phase=phase,
                 python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],version=1)
    torch.save(payload,str(path)+'.tmp'); os.replace(str(path)+'.tmp',path)
def load(path,model,opt=None,restore_rng=False):
    # Only load trusted local checkpoints (contains Python/NumPy RNG state).
    ck=torch.load(path,map_location='cpu',weights_only=False); model.load_state_dict(ck['model'])
    if opt is not None: opt.load_state_dict(ck['optimizer'])
    if restore_rng:
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck
