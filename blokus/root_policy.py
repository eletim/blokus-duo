"""Root-only distillation. Candidate membership and opponent ranking stay fixed."""
import os,time,random
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from .ai import features,heuristic,seed_all
from .engine import legal_moves,play
from .research import clone
from .value_search import BudgetAgent

EXTRA=27
XDIM=12+EXTRA

def inputs(s,qs,x=None,st=None):
    if x is None:x,st=features(s,qs)
    extra=np.zeros((len(qs),EXTRA),np.float32)
    for i,q in enumerate(qs):
        cells=q.move.cells;rs=[r for r,c in cells];cs=[c for r,c in cells]
        extra[i,q.move.piece]=1
        extra[i,21:]=[sum(rs)/len(rs)/13,sum(cs)/len(cs)/13,min(rs)/13,min(cs)/13,(max(rs)-min(rs))/4,(max(cs)-min(cs))/4]
    state=np.r_[st,np.array([s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],np.float32)]
    return np.concatenate((x,extra),axis=1),state.astype(np.float32)

class RootPolicy(nn.Module):
    def __init__(self):
        super().__init__();self.state=nn.Sequential(nn.Linear(49,32),nn.Tanh())
        self.head=nn.Sequential(nn.Linear(XDIM+32,64),nn.Tanh(),nn.Dropout(.1),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,x,st):
        f=self.state(st).unsqueeze(-2).expand(*x.shape[:-1],32)
        return self.head(torch.cat((x,f),-1)).squeeze(-1)

def masked_loss(logits,mask,target):
    if not bool(mask.gather(1,target[:,None]).all()):raise ValueError('target must be evaluated and legal')
    return nn.functional.cross_entropy(logits.masked_fill(~mask,-1e9),target)

class StudentAgent(BudgetAgent):
    def __init__(self,checkpoint,budget_ms=6,policy_path=None,device='cuda'):
        super().__init__('value',checkpoint,.5,24,budget_ms,device)
        self.policy=None;self.decisions=[]
        if policy_path:
            ck=torch.load(policy_path,map_location='cpu',weights_only=False);self.policy=RootPolicy().to(device);self.policy.load_state_dict(ck['model']);self.policy.eval()
            with torch.no_grad():self.policy(torch.zeros(128,XDIM,device=device),torch.zeros(49,device=device))
            if device.startswith('cuda'):torch.cuda.synchronize()
    def act(self,s,rng):
        if s.done:raise ValueError('terminal')
        start=time.perf_counter();deadline=start+self.budget_ms/1000
        qs=legal_moves(s);generation=time.perf_counter()-start;rank_start=time.perf_counter();policy_ms=0.
        if not qs:
            self.times.append(time.perf_counter()-start);self.decisions.append(dict(ms=self.times[-1]*1000,policy_ms=0.,generation_ms=generation*1000,completed_roots=0,candidates=0));return None
        x,st=features(s,qs);prior=heuristic(x);members=np.argsort(-prior,kind='stable')[:128];order=members
        if self.policy is not None:
            t=time.perf_counter();xx,ss=inputs(s,[qs[int(i)] for i in members],x[members],st)
            with torch.no_grad():residual=self.policy(torch.as_tensor(xx,device=self.device),torch.as_tensor(ss,device=self.device)).cpu().numpy()
            # cpu() synchronizes GPU; input construction, transfer and synchronization charged.
            order=members[np.argsort(-(prior[members]+residual),kind='stable')];policy_ms=(time.perf_counter()-t)*1000
        self.sections['legal']+=generation;self.sections['rank']+=time.perf_counter()-rank_start
        values=[];done=[]
        for i in order:
            if time.perf_counter()+self.cost_estimate*1.2+.0001>=deadline:break
            step=time.perf_counter();c=clone(s);play(c,qs[int(i)].move,validate=False);self.nodes+=1
            replies=legal_moves(c);leaves=[]
            if replies:
                rx,_=features(c,replies)
                for j in np.argsort(-heuristic(rx),kind='stable')[:3]:
                    leaf=clone(c);play(leaf,replies[int(j)].move,validate=False);leaves.append(leaf);self.nodes+=1
            else:
                leaf=clone(c);play(leaf,None,validate=False);leaves=[leaf];self.nodes+=1
            t=time.perf_counter();self.sections['expand']+=t-step;v=float(self.evaluator.batch(leaves).min());self.sections['value']+=time.perf_counter()-t
            cost=time.perf_counter()-step;cap=self.budget_ms/1000
            self.cost_estimate=max(.00015,min(cap*.45,.75*self.cost_estimate+.25*min(cost,cap)))
            if time.perf_counter()>deadline:break
            done.append(int(i));values.append(v);self.roots+=1
        if done:
            vv=np.asarray(values);choice=done[int(rng.choice(np.flatnonzero(vv>=vv.max()-1e-7)))]
        else:choice=int(order[0]);self.fallbacks+=1
        self.times.append(time.perf_counter()-start);self.decisions.append(dict(ms=self.times[-1]*1000,policy_ms=policy_ms,generation_ms=generation*1000,completed_roots=len(done),candidates=len(members)))
        return qs[choice].move
