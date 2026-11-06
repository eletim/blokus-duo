"""Matched-experience linear policy controls plus separate Monte Carlo value fit."""
import argparse,copy,hashlib,json,random
from pathlib import Path
import numpy as np
from .engine import State,play,legal_moves
from .linear_rl import prefix
from .linear_value import Agent,value_features,outcome

def state_hash(s):
 return hashlib.sha256(json.dumps([s.boards,s.remaining,[s.score(0),s.score(1)],s.turn,s.passes],separators=(',',':')).encode()).hexdigest()

def adam(weights,grad,m,v,step,lr):
 norm=float(np.linalg.norm(grad));grad=grad*min(1.,5/max(norm,1e-12));m*=.9;m+=.1*grad;v*=.999;v+=.001*grad*grad
 weights+=lr*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
 if not np.isfinite(weights).all() or np.abs(weights).max()>1000:raise FloatingPointError('linear training collapsed')
 return norm

class Trainer:
 def __init__(self,seed,mode='new',backend='native',series=2110000000,forbidden=None):
  self.seed=seed;self.mode=mode;self.backend=backend;self.series=series;self.forbidden=set(forbidden or []);self.iteration=0;self.rng=np.random.default_rng(seed)
  self.agent=Agent(mode,backend=backend);self.initial=Agent(mode,backend=backend);self.past=self.agent.weights.copy();self.m=np.zeros_like(self.past);self.v=self.m.copy();self.vm=np.zeros((3,8));self.vv=self.vm.copy()
 def step(self,visited=None):
  current=Agent(self.mode,self.agent.weights,backend=self.backend);past=Agent(self.mode,self.past,backend=self.backend);grad=np.zeros_like(self.m);vphi=[];targets=[];games=[];entropies=[];vpred=[];rewards=[]
  for game in range(16):
   seed=self.series+self.seed*100000+self.iteration*16+game;retry=0
   while True:
    ps=seed+retry*100000000000;s=prefix(ps)
    if state_hash(s) not in self.forbidden:break
    retry+=1
   seat=game%2;kind=(game//2)%4;opponent=current if kind<2 else past if kind==2 else self.initial;trajectory=np.zeros_like(grad);phis=[];moves=[];learner_states=0
   while not s.done:
    if visited is not None:visited.add(state_hash(s))
    if s.turn==seat:
     if self.mode=='new':phis.append(value_features(s,self.backend))
     m,g,entropy=self.agent.sample(s,self.rng);trajectory+=g;entropies.append(entropy);learner_states+=1
    elif kind==3:m=opponent.act(s,self.rng)
    else:m,_,_=opponent.sample(s,self.rng)
    moves.append(None if m is None else [m.piece,m.orientation,m.row,m.col]);play(s,m,validate=False)
   if visited is not None:visited.add(state_hash(s))
   reward=outcome(s,seat);grad+=reward*trajectory;rewards.append(reward)
   vphi.extend(phis);targets.extend([reward]*len(phis));games.append(dict(prefix_seed=ps,seat=seat,kind='current' if kind<2 else 'past' if kind==2 else 'initial',moves=moves,scores=[s.score(0),s.score(1)],reward=reward,learner_states=learner_states,rejected_starts=retry))
  self.iteration+=1;norm=adam(self.agent.weights,grad/16,self.m,self.v,self.iteration,.03)
  mse=None
  if vphi:
   phi=np.asarray(vphi);y=np.array(targets);pred=np.sum(phi*self.agent.value_weights,axis=(1,2));mse=float(np.mean((pred-y)**2));vg=np.mean((y-pred)[:,None,None]*phi,axis=0);adam(self.agent.value_weights,vg,self.vm,self.vv,self.iteration,.01)
  if self.iteration%64==0:self.past=self.agent.weights.copy()
  row=dict(iteration=self.iteration,mean_reward=float(np.mean(rewards)),gradient_norm=norm,entropy=float(np.mean(entropies)),value_mse=mse,opponents={k:dict(games=sum(g['kind']==k for g in games),learner_states=sum(g['learner_states'] for g in games if g['kind']==k),reward=float(np.mean([g['reward'] for g in games if g['kind']==k]))) for k in ('current','past','initial')})
  return row,games
 def save(self,path):
  d=dict(format='linear-outcome-trainer-v2',seed=self.seed,mode=self.mode,series=self.series,iteration=self.iteration,rng=self.rng.bit_generator.state,forbidden=sorted(self.forbidden),weights=self.agent.weights.tolist(),value=self.agent.value_weights.tolist(),**{k:getattr(self,k).tolist() for k in ('past','m','v','vm','vv')})
  Path(path).write_text(json.dumps(d)+'\n')
 @classmethod
 def load(cls,path,backend='native'):
  d=json.loads(Path(path).read_text());assert d['format']=='linear-outcome-trainer-v2';t=cls(d['seed'],d['mode'],backend,d['series'],d['forbidden']);t.iteration=d['iteration'];t.rng.bit_generator.state=d['rng'];t.agent=Agent(d['mode'],d['weights'],d['value'],backend)
  for k in ('past','m','v','vm','vv'):setattr(t,k,np.array(d[k]))
  return t
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resume');p.add_argument('--seed',type=int,default=3101);p.add_argument('--mode',choices=['old','new'],default='new');p.add_argument('--until',type=int,default=64);p.add_argument('--backend',default='native',choices=['native','python']);a=p.parse_args();root=Path(a.output);root.mkdir(exist_ok=True,parents=True)
 if (root/'state.json').exists() and not a.resume:p.error('existing checkpoint; explicitly resume')
 t=Trainer.load(a.resume,a.backend) if a.resume else Trainer(a.seed,a.mode,a.backend)
 while t.iteration<a.until:
  row,_=t.step()
  if t.iteration%16==0:print(json.dumps(row),flush=True)
 t.save(root/'state.json');t.agent.save(root/'policy.json',dict(seed=t.seed,iteration=t.iteration))
