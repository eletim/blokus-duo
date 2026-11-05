"""Isolated reply-width ablation; original BudgetAgent remains unchanged."""
import time
import numpy as np
import torch
from .value_search import BudgetAgent
from .engine import legal_moves,play
from .ai import features,heuristic
from .research import clone,pack,to_device


def reset_game(agent):
    """Reset search/adaptive scheduling state, retain model and cumulative counters."""
    if hasattr(agent,'tree'):agent.tree.tt.clear()
    if hasattr(agent,'old'):reset_game(agent.old)
    agent.cost_estimate=.0008 if agent.kind=='value' else .0006
    agent.batch_reserve=.0018 if agent.kind=='value' else .0005
    agent.expand_estimate=.0005


class ReplyAgent(BudgetAgent):
    def __init__(self,*args,reply_width=6,**kwargs):
        if reply_width<1:raise ValueError('positive reply width required')
        super().__init__(*args,**kwargs);self.reply_width=reply_width
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
                rx,_=features(child,replies);indices=np.argsort(-heuristic(rx),kind='stable')[:self.reply_width]
                leaves=[]
                for j in indices:
                    leaf=clone(child);play(leaf,replies[int(j)].move,validate=False);leaves.append(leaf);self.nodes+=1
            else:
                leaf=clone(child);play(leaf,None,validate=False);leaves=[leaf];self.nodes+=1
            self.sections['expand']+=time.perf_counter()-t;t=time.perf_counter()
            # Two plies: leaf is in root player's perspective; opponent minimizes.
            value=float(self.evaluator.batch(leaves).min())
            self.sections['value']+=time.perf_counter()-t
            cost=time.perf_counter()-step
            # A rare OS/GPU stall must not make all future children unaffordable.
            # Clip the observation and bound the estimate below one turn's budget.
            cap=self.budget_ms/1000
            self.cost_estimate=max(.00015,min(cap*.45,.75*self.cost_estimate+.25*min(cost,cap)))
            # Discard a child that finished after the deadline rather than using late work.
            if time.perf_counter()>deadline:break
            done.append(int(i));values.append(value);self.roots+=1
        if done:
            v=np.asarray(values);best=np.flatnonzero(v>=v.max()-1e-7);choice=done[int(rng.choice(best))]
        else:self.fallbacks+=1;choice=int(order[0])
        self.times.append(time.perf_counter()-start)
        return qs[choice].move
