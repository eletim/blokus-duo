"""NumPy-only policy math: phase-specific linear REINFORCE, no neural model."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
from .engine import PLACEMENTS, State, legal_moves, play

ACTIVE = [0,1,2,3,4,5,6,7,8,11]  # drop action-independent remaining/difference
PRIOR = np.array([12,4,2,1.2,2,.3,1,.2,-.6,.2], dtype=np.float64)
FORMAT = 'phase-linear-reinforce-v1'


def inputs(s, backend='python'):
    if backend == 'native':
        from .ppo_native import scan
        ids, x = scan(s)
        qs = [PLACEMENTS[i] for i in ids]
    elif backend == 'python':
        from .ai import features
        qs = legal_moves(s)
        x, _ = features(s, qs)
    else:
        raise ValueError('backend must be python or native')
    remaining = s.remaining[s.turn].bit_count()
    phase = 0 if remaining > 14 else 1 if remaining > 7 else 2
    return np.asarray(x[:, ACTIVE], dtype=np.float64), qs, phase


def probabilities(x, weights):
    logits = x @ weights
    p = np.exp(logits-logits.max())
    return p/p.sum()


class LinearAgent:
    def __init__(self, weights=None, backend='python'):
        self.weights = np.tile(PRIOR,(3,1)) if weights is None else np.array(weights,dtype=np.float64)
        if self.weights.shape != (3,10) or not np.isfinite(self.weights).all():
            raise ValueError('invalid weights')
        self.backend = backend

    def act(self, s, rng):
        x, qs, phase = inputs(s, self.backend)
        if not qs:
            return None
        scores = x @ self.weights[phase]
        return qs[int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)))].move

    def sample(self,s,rng):
        x, qs, phase = inputs(s,self.backend)
        grad = np.zeros((3,10))
        if not qs:
            return None, grad
        p = probabilities(x,self.weights[phase])
        i = int(rng.choice(len(qs),p=p))
        grad[phase] = x[i]-p@x
        return qs[i].move, grad

    def save(self,path,metadata=None):
        Path(path).write_text(json.dumps(dict(format=FORMAT,weights=self.weights.tolist(),metadata=metadata or {}),indent=2))

    @classmethod
    def load(cls,path,backend='python'):
        data=json.loads(Path(path).read_text())
        if data['format']!=FORMAT:raise ValueError('unknown format')
        return cls(data['weights'],backend)


def prefix(seed):
    rng=random.Random(seed)
    s=State()
    for _ in range(4):
        qs=legal_moves(s)
        play(s,rng.choice(qs).move if qs else None,validate=False)
    return s


def state_key(s):
    return str((s.boards,s.remaining,s.last,s.turn,s.passes))


class Trainer:
    def __init__(self,seed,backend='python',lr=.03):
        self.agent=LinearAgent(backend=backend)
        self.initial=LinearAgent(backend=backend)
        self.past=self.agent.weights.copy()
        self.rng=np.random.default_rng(seed)
        self.seed=seed; self.iteration=0; self.lr=lr
        self.m=np.zeros((3,10)); self.v=self.m.copy()

    def step(self,batch=16,visited=None):
        # Freeze all opponents and learner weights throughout this batch.
        current=LinearAgent(self.agent.weights.copy(),self.agent.backend)
        past=LinearAgent(self.past,self.agent.backend)
        gradient=np.zeros((3,10)); rewards=[]
        for game in range(batch):
            s=prefix(970000000+self.seed*100000+self.iteration*batch+game)
            seat=game%2; kind=(game//2)%4
            opponent=current if kind<2 else past if kind==2 else self.initial
            trajectory=np.zeros((3,10))
            while not s.done:
                if visited is not None:visited.add(state_key(s))
                if s.turn==seat:
                    move,g=self.agent.sample(s,self.rng);trajectory+=g
                elif kind==3:
                    move=opponent.act(s,self.rng)
                else:
                    move,_=opponent.sample(s,self.rng)
                play(s,move,validate=False)
            difference=s.score(seat)-s.score(1-seat)
            reward=float((difference>0)-(difference<0))
            gradient+=reward*trajectory
            rewards.append(reward)
        gradient/=batch
        norm=float(np.linalg.norm(gradient))
        gradient*=min(1.,5/max(norm,1e-12))
        self.iteration+=1
        self.m=.9*self.m+.1*gradient
        self.v=.999*self.v+.001*gradient*gradient
        self.agent.weights+=self.lr*(self.m/(1-.9**self.iteration))/(np.sqrt(self.v/(1-.999**self.iteration))+1e-8)
        if not np.isfinite(self.agent.weights).all():raise FloatingPointError('nonfinite update')
        if self.iteration%64==0:self.past=self.agent.weights.copy()
        return dict(iteration=self.iteration,mean_reward=float(np.mean(rewards)),gradient_norm=norm)

    def save(self,path):
        data=dict(format=FORMAT,weights=self.agent.weights.tolist(),past=self.past.tolist(),m=self.m.tolist(),v=self.v.tolist(),iteration=self.iteration,lr=self.lr,seed=self.seed,rng=self.rng.bit_generator.state)
        path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data));tmp.replace(path)

    @classmethod
    def load(cls,path,backend='python'):
        d=json.loads(Path(path).read_text())
        if d['format']!=FORMAT:raise ValueError('unknown format')
        obj=cls(d['seed'],backend,d['lr']);obj.agent=LinearAgent(d['weights'],backend)
        for k in ('past','m','v'):setattr(obj,k,np.array(d[k]))
        obj.iteration=d['iteration'];obj.rng.bit_generator.state=d['rng']
        return obj


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True);parser.add_argument('--iterations',type=int,default=256)
    parser.add_argument('--seed',type=int,default=3101);parser.add_argument('--backend',choices=['python','native'],default='python');parser.add_argument('--resume')
    args=parser.parse_args()
    if args.iterations<=0:parser.error('iterations must be positive')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'state.json').exists() and not args.resume:parser.error('existing run; use --resume')
    t=Trainer.load(args.resume,args.backend) if args.resume else Trainer(args.seed,args.backend)
    for _ in range(args.iterations):
        row=t.step()
        if t.iteration%16==0:print(json.dumps(row),flush=True)
    t.save(root/'state.json');t.agent.save(root/'policy.json',dict(iteration=t.iteration,seed=t.seed))

if __name__=='__main__':main()
