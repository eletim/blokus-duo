"""Value-only Monte Carlo fitting with a frozen non-neural behavior policy."""
import argparse,copy,json,random
from pathlib import Path
import numpy as np
from .linear_value import Agent,Search,value_features,outcome
from .linear_train2 import adam,state_hash
from .linear_rl import prefix
from .engine import play

def tuples(x):return tuple(tuples(v) for v in x) if isinstance(x,list) else x
class ValueTrainer:
 def __init__(self,seed,source,arm='P',backend='native',series=2290000000,forbidden=None):
  if arm not in ('P','S'):raise ValueError('arm must be P or S')
  if source.mode!='old':raise ValueError('fixed old policy required')
  self.seed=seed;self.arm=arm;self.backend=backend;self.series=series;self.forbidden=set(forbidden or []);self.rng=random.Random(2320000000+seed);self.iteration=0
  self.frozen=Agent('old',source.weights,source.value_weights,backend);self.agent=Agent('old',source.weights,source.value_weights,backend);self.initial=Agent('old',backend=backend)
  self.m=np.zeros((3,8));self.v=self.m.copy()
 def step(self,visited=None):
  bot=Search(self.frozen,ms=0) if self.arm=='S' else self.frozen;phis=[];targets=[];games=[]
  for i in range(16):
   ps=self.series+self.seed*100000+self.iteration*16+i;retries=0
   while state_hash(prefix(ps+retries*100000000000)) in self.forbidden:retries+=1
   ps+=retries*100000000000;s=prefix(ps);seat=i%2;kind='initial' if (i//2)%4==3 else 'fixed';moves=[];local=[]
   while not s.done:
    if visited is not None:visited.add(state_hash(s))
    if s.turn==seat:local.append(value_features(s,self.backend))
    actor=self.initial if s.turn!=seat and kind=='initial' else bot;m=actor.act(s,self.rng);moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   if visited is not None:visited.add(state_hash(s))
   y=outcome(s,seat);phis.extend(local);targets.extend([y]*len(local));games.append(dict(prefix_seed=ps,seat=seat,kind=kind,moves=moves,scores=[s.score(0),s.score(1)],reward=y,learner_states=len(local),rejected_starts=retries))
  phi=np.asarray(phis);y=np.array(targets);pred=np.sum(phi*self.agent.value_weights,axis=(1,2));grad=np.mean((y-pred)[:,None,None]*phi,axis=0);self.iteration+=1;norm=adam(self.agent.value_weights,grad,self.m,self.v,self.iteration,.01)
  assert np.array_equal(self.agent.weights,self.frozen.weights)
  return dict(iteration=self.iteration,mse=float(np.mean((pred-y)**2)),gradient_norm=norm,states=len(y),max_abs_value_weight=float(np.abs(self.agent.value_weights).max()),mean_reward=float(np.mean([g['reward'] for g in games]))),games
 def save(self,path):
  d=dict(format='fixed-linear-value-fit-v1',seed=self.seed,arm=self.arm,series=self.series,iteration=self.iteration,forbidden=sorted(self.forbidden),rng=self.rng.getstate(),policy=self.frozen.weights.tolist(),generator_value=self.frozen.value_weights.tolist(),value=self.agent.value_weights.tolist(),m=self.m.tolist(),v=self.v.tolist(),lr=.01)
  Path(path).write_text(json.dumps(d)+'\n')
 @classmethod
 def load(cls,path,backend='native'):
  d=json.loads(Path(path).read_text())
  if d.get('format')!='fixed-linear-value-fit-v1' or d['lr']!=.01:raise ValueError('checkpoint format/LR mismatch')
  source=Agent('old',d['policy'],d['generator_value'],backend);t=cls(d['seed'],source,d['arm'],backend,d['series'],d['forbidden']);t.iteration=d['iteration'];t.rng.setstate(tuples(d['rng']));t.agent.value_weights[:]=d['value'];t.m[:]=d['m'];t.v[:]=d['v'];return t
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source');p.add_argument('--resume');p.add_argument('--output',required=True);p.add_argument('--until',type=int,required=True);p.add_argument('--seed',type=int,default=3101);p.add_argument('--arm',choices=['P','S'],default='P');a=p.parse_args()
 t=ValueTrainer.load(a.resume) if a.resume else ValueTrainer(a.seed,Agent.load(a.source),a.arm)
 while t.iteration<a.until:t.step()
 t.save(a.output)
