"""Outcome value learning and shared, measured wall-budget beam search."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
import random
import numpy as np
import torch
from torch import nn
from .engine import State, START, cells_of, frontier, legal_moves, play
from .ai import features, heuristic
from .research import clone, Agent, pack, to_device

SCALE=40.
SWAP_FEATURES=list(range(21,42))+list(range(21))+[43,42,45,44,47,46,48]


def encode(s):
    board=np.zeros((6,14,14),dtype=np.uint8)
    fs=[frontier(s,p)[0] for p in (s.turn,1-s.turn)]
    for i,p in enumerate((s.turn,1-s.turn)):
        for r,c in cells_of(s.boards[p]):board[i,r,c]=1
        for r,c in cells_of(fs[i]):board[i+2,r,c]=1
        board[i+4,START[p][0],START[p][1]]=1
    st=np.array([float(s.remaining[p]>>i&1) for p in (s.turn,1-s.turn) for i in range(21)]+
                [s.boards[s.turn].bit_count()/89,s.boards[1-s.turn].bit_count()/89,fs[0].bit_count()/40,fs[1].bit_count()/40,
                 s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],dtype=np.float32)
    return board,st


def swap_inputs(board,st):return board[:,[1,0,3,2,5,4]],st[:,SWAP_FEATURES]


def base_value(st):return (st[:,46]-st[:,47])*89/SCALE+.6*(st[:,44]-st[:,45])


def exact_terminal(s):
    return s.done or (s.remaining==[0,0]) or (s.passes>0 and not legal_moves(s))


def score_value(s):return (s.score(s.turn)-s.score(1-s.turn))/SCALE


class OutcomeNet(nn.Module):
    def __init__(self,arch='flat'):
        super().__init__();self.arch=arch
        if arch not in ('flat','pooled'):raise ValueError('unknown architecture')
        self.conv=nn.Sequential(nn.Conv2d(6,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,24,3,padding=1),nn.ReLU())
        self.image=(nn.Sequential(nn.Flatten(),nn.Linear(24*14*14,64),nn.ReLU()) if arch=='flat' else
                    nn.Sequential(nn.AvgPool2d(7,7),nn.Flatten(),nn.Linear(24*4,32),nn.ReLU()))
        self.head=nn.Sequential(nn.Linear((64 if arch=='flat' else 32)+49,96),nn.ReLU(),nn.Linear(96,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,board,st):
        other_board,other_st=swap_inputs(board,st)
        b=torch.cat((board,other_board));x=torch.cat((st,other_st))
        f=self.head(torch.cat((self.image(self.conv(b)),x),1)).squeeze(1)
        # Enforce exact colour/perspective antisymmetry, preserving last-piece bonus.
        residual=(f[:len(st)]-f[len(st):])/2
        return base_value(st)+residual


class ValueEvaluator:
    def __init__(self,path=None,device='cuda',alpha=1.,gate=42):
        self.device=device;self.alpha=alpha;self.gate=gate;self.model=None
        if path:
            ck=torch.load(path,map_location='cpu',weights_only=False)
            if ck.get('format')!=4:raise ValueError('not an outcome checkpoint')
            self.model=OutcomeNet(ck.get('arch','flat')).to(device);self.model.load_state_dict(ck['model']);self.model.eval()
            # Model initialization/warm-up is not charged to game decisions.
            with torch.no_grad():self.model(torch.zeros(3,6,14,14,device=device),torch.zeros(3,49,device=device))
            if device.startswith('cuda'):torch.cuda.synchronize()
    def batch(self,states):
        values=[];learn=[];indices=[]
        for i,s in enumerate(states):
            if exact_terminal(s):values.append(score_value(s));continue
            f0=frontier(s,s.turn)[0].bit_count();f1=frontier(s,1-s.turn)[0].bit_count()
            v=score_value(s)+.6*(f0-f1)/SCALE;values.append(v)
            if self.model is not None and self.alpha and sum(p.bit_count() for p in s.remaining)<=self.gate:
                learn.append(encode(s));indices.append(i)
        if learn:
            b=torch.as_tensor(np.stack([a for a,x in learn]),dtype=torch.float32,device=self.device)
            x=torch.as_tensor(np.stack([x for a,x in learn]),device=self.device)
            with torch.no_grad():pred=self.model(b,x).cpu().numpy()
            for i,p in zip(indices,pred):values[i]=(1-self.alpha)*values[i]+self.alpha*float(p)
        return np.array(values)


class BudgetAgent:
    """Same allowance, not padded equal elapsed time. Complete beam children only.

    The budget starts before legal generation and includes rank/encoding/GPU transfer.
    Atomic operations may overrun; all elapsed times and overrun flags are recorded.
    Expensive next children are refused using an online conservative cost estimate.
    """
    def __init__(self,kind='static',checkpoint=None,alpha=1.,gate=42,budget_ms=6.,device='cuda',max_roots=128,scheduler='incremental'):
        if budget_ms<=0:raise ValueError('positive budget required')
        self.kind=kind;self.budget_ms=budget_ms;self.max_roots=max_roots;self.scheduler=scheduler
        self.evaluator=ValueEvaluator(checkpoint if kind=='value' else None,device,alpha,gate)
        self.rank=Agent('artifacts/research-round1/C/selected.pt',device) if kind=='C' else None
        if self.rank:
            with torch.no_grad():self.rank.model(*to_device(pack(State(),legal_moves(State())),device))
            if device.startswith('cuda'):torch.cuda.synchronize()
        self.device=device;self.times=[];self.nodes=0;self.roots=0;self.fallbacks=0
        self.sections=dict(legal=0.,rank=0.,expand=0.,value=0.)
        self.cost_estimate=.0008 if kind=='value' else .0006
        self.batch_reserve=.0018 if kind=='value' else .0005
        self.expand_estimate=.0005
    def act(self,s,rng):
        if self.scheduler=='batched':return self.act_batched(s,rng)
        if s.done:raise ValueError('terminal state')
        start=time.perf_counter();deadline=start+self.budget_ms/1000;t=start
        qs=legal_moves(s);self.sections['legal']+=time.perf_counter()-t
        if not qs:
            self.times.append(time.perf_counter()-start);return None
        t=time.perf_counter();x,st=features(s,qs);scores=heuristic(x)
        if self.rank and time.perf_counter()+.0005<deadline:
            with torch.no_grad():scores=self.rank.model(*to_device(pack(s,qs),self.device))[0].cpu().numpy()
        self.sections['rank']+=time.perf_counter()-t
        order=np.argsort(-scores,kind='stable')[:self.max_roots];values=[];done=[]
        for i in order:
            if time.perf_counter()+self.cost_estimate*1.2+.0001>=deadline:break
            step=time.perf_counter();t=step
            child=clone(s);play(child,qs[int(i)].move,validate=False);self.nodes+=1
            replies=legal_moves(child)
            if replies:
                rx,_=features(child,replies);indices=np.argsort(-heuristic(rx),kind='stable')[:3]
                leaves=[]
                for j in indices:
                    leaf=clone(child);play(leaf,replies[int(j)].move,validate=False);leaves.append(leaf);self.nodes+=1
            else:
                leaf=clone(child);play(leaf,None,validate=False);leaves=[leaf];self.nodes+=1
            self.sections['expand']+=time.perf_counter()-t;t=time.perf_counter()
            # Two plies: leaf is in root player's perspective; opponent minimizes.
            value=float(self.evaluator.batch(leaves).min())
            self.sections['value']+=time.perf_counter()-t
            cost=time.perf_counter()-step;self.cost_estimate=max(.00015,.75*self.cost_estimate+.25*cost)
            # Discard a child that finished after the deadline rather than using late work.
            if time.perf_counter()>deadline:break
            done.append(int(i));values.append(value);self.roots+=1
        if done:
            v=np.asarray(values);best=np.flatnonzero(v>=v.max()-1e-7);choice=done[int(rng.choice(best))]
        else:self.fallbacks+=1;choice=int(order[0])
        self.times.append(time.perf_counter()-start)
        return qs[choice].move
    def act_batched(self,s,rng):
        if s.done:raise ValueError('terminal state')
        start=time.perf_counter();deadline=start+self.budget_ms/1000;t=start
        qs=legal_moves(s);self.sections['legal']+=time.perf_counter()-t
        if not qs:self.times.append(time.perf_counter()-start);return None
        t=time.perf_counter();x,_=features(s,qs);scores=heuristic(x)
        if self.rank and time.perf_counter()+.0005<deadline:
            with torch.no_grad():scores=self.rank.model(*to_device(pack(s,qs),self.device))[0].cpu().numpy()
        self.sections['rank']+=time.perf_counter()-t
        order=np.argsort(-scores,kind='stable')[:self.max_roots];leaves=[];groups=[]
        # Strong heuristic fallback remains usable if the final GPU batch is late.
        for i in order:
            if time.perf_counter()+1.2*self.expand_estimate+self.batch_reserve+.0001>=deadline:break
            t=time.perf_counter();child=clone(s);play(child,qs[int(i)].move,validate=False);self.nodes+=1
            replies=legal_moves(child);begin=len(leaves)
            if replies:
                rx,_=features(child,replies)
                for j in np.argsort(-heuristic(rx),kind='stable')[:3]:
                    leaf=clone(child);play(leaf,replies[int(j)].move,validate=False);leaves.append(leaf);self.nodes+=1
            else:
                leaf=clone(child);play(leaf,None,validate=False);leaves.append(leaf);self.nodes+=1
            groups.append((int(i),begin,len(leaves)))
            cost=time.perf_counter()-t;self.sections['expand']+=cost;self.expand_estimate=.75*self.expand_estimate+.25*cost
        choice=int(order[0])
        if leaves and time.perf_counter()<deadline:
            t=time.perf_counter();v=self.evaluator.batch(leaves);cost=time.perf_counter()-t
            self.sections['value']+=cost;self.batch_reserve=max(.0002,.7*self.batch_reserve+.3*cost*1.3)
            if time.perf_counter()<=deadline:
                scores=np.array([v[a:b].min() for _,a,b in groups]);best=np.flatnonzero(scores>=scores.max()-1e-7)
                choice=groups[int(rng.choice(best))][0];self.roots+=len(groups)
            else:self.fallbacks+=1
        else:self.fallbacks+=1
        self.times.append(time.perf_counter()-start);return qs[choice].move

    def stats(self):
        t=np.asarray(self.times)*1000
        return dict(calls=len(t),milliseconds_per_move=float(t.mean()) if len(t) else 0,p50_ms=float(np.median(t)) if len(t) else 0,
                    p95_ms=float(np.quantile(t,.95)) if len(t) else 0,max_ms=float(t.max()) if len(t) else 0,
                    allowance_ms=self.budget_ms,overruns=int((t>self.budget_ms).sum()),overrun_ms=float(np.maximum(t-self.budget_ms,0).sum()),
                    total_seconds=float(t.sum()/1000),nodes=self.nodes,completed_roots=self.roots,fallbacks=self.fallbacks,sections=self.sections)
