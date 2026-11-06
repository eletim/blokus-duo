"""CPU/NumPy local pattern residual value, terminal WDL targets, no neural layers."""
import json,copy
from pathlib import Path
from functools import lru_cache
import numpy as np
from .linear_value import Agent,value_features,outcome
from .linear_train2 import adam
FORMAT='linear-plus-local-tuple-v1'
SIZE=3*2*729

def phase(s):
 n=(s.remaining[0].bit_count()+s.remaining[1].bit_count())/2
 return 0 if n>14 else 1 if n>7 else 2

@lru_cache(maxsize=1)
def geometry():
 windows=[];groups=[]
 for h,w in ((2,3),(3,2)):
  for r in range(15-h):
   for c in range(15-w):
    # Transpose 3x2 into the same 2x3 ordering.
    cells=[(r+i,c+j) for i in range(h) for j in range(w)] if h==2 else [(r+j,c+i) for i in range(w) for j in range(h)]
    windows.append([x*14+y for x,y in cells]);groups.append(int(r==0 or c==0 or r+h==14 or c+w==14))
 powers=3**np.arange(6);digits=(np.arange(729)[:,None]//powers)%3
 canonical=np.minimum(np.arange(729),digits[:,::-1]@powers)
 swapdigits=np.where(digits==0,0,3-digits);swap=canonical[swapdigits@powers]
 swap_all=np.concatenate([swap+729*i for i in range(6)])
 arrays=(np.array(windows,dtype=np.int64),np.array(groups,dtype=np.int64),powers,canonical,swap_all)
 for x in arrays:x.flags.writeable=False
 assert len(windows)==312
 return arrays

def codes(s):
 windows,groups,powers,canonical,_=geometry()
 bits=[np.unpackbits(np.frombuffer(int(s.boards[p]).to_bytes(25,'little'),dtype=np.uint8),bitorder='little')[:196] for p in (s.turn,1-s.turn)]
 board=bits[0]+2*bits[1];raw=board[windows]@powers
 return (canonical[raw]+729*groups+1458*phase(s)).astype(np.uint16)

def residual(table,ids):
 swap=geometry()[4]
 return np.mean(table[ids]-table[swap[ids]],axis=-1)/2

def table_gradient(ids,errors):
 swap=geometry()[4];scale=2*ids.shape[0]*ids.shape[1];w=np.repeat(errors,ids.shape[1])/scale
 return np.bincount(ids.ravel(),weights=w,minlength=SIZE)-np.bincount(swap[ids].ravel(),weights=w,minlength=SIZE)

class TupleAgent(Agent):
 def __init__(self,policy,base_value,table=None,backend='native'):
  super().__init__('old',policy,base_value,backend)
  self.table=np.zeros(SIZE,dtype=np.float64) if table is None else np.asarray(table,dtype=np.float64).copy()
  if self.table.shape!=(SIZE,) or not np.isfinite(self.table).all():raise ValueError('invalid tuple table')
 def value(self,s):
  if s.done:return outcome(s,s.turn)
  return float(np.clip(np.sum(self.value_weights*value_features(s,self.backend))+residual(self.table,codes(s)),-1,1))
 def save(self,path,metadata=None):
  Path(path).write_text(json.dumps(dict(format=FORMAT,policy=self.weights.tolist(),base_value=self.value_weights.tolist(),table=self.table.tolist(),metadata=metadata or {}),indent=2)+'\n')
 @classmethod
 def load(cls,path,backend='native'):
  d=json.loads(Path(path).read_text())
  if d.get('format')!=FORMAT:raise ValueError('wrong tuple model format')
  return cls(d['policy'],d['base_value'],d['table'],backend)

def load_value(path,backend='native'):
 d=json.loads(Path(path).read_text())
 return TupleAgent.load(path,backend) if d.get('format')==FORMAT else Agent.load(path,backend)

class Learner:
 def __init__(self,source,arm,seed,epochs=8,batch=256):
  if arm not in ('L','T'):raise ValueError('L/T')
  self.arm=arm;self.policy=source.weights.copy();self.base=source.value_weights.copy();self.w=self.base.reshape(-1).copy() if arm=='L' else np.zeros(SIZE)
  self.m=np.zeros_like(self.w);self.v=self.m.copy();self.step=0;self.epoch=0;self.cursor=0;self.order=None;self.rng=np.random.default_rng(seed);self.epochs=epochs;self.batch=batch;self.visits=np.zeros(SIZE,dtype=np.int64)
 def update(self,phi,ids,y):
  if self.epoch>=self.epochs:return None
  if self.order is None:self.order=self.rng.permutation(len(y));self.cursor=0
  ix=self.order[self.cursor:self.cursor+self.batch];p=phi[ix];c=ids[ix];target=y[ix]
  pred=p@self.w if self.arm=='L' else p@self.base.reshape(-1)+residual(self.w,c)
  error=target-pred;grad=np.mean(error[:,None]*p,axis=0) if self.arm=='L' else table_gradient(c,error)
  self.step+=1;adam(self.w,grad,self.m,self.v,self.step,.01)
  self.visits+=np.bincount(c.ravel(),minlength=SIZE)
  self.cursor+=len(ix)
  if self.cursor==len(y):self.epoch+=1;self.order=None;self.cursor=0
  return dict(step=self.step,epoch=self.epoch,mse=float(np.mean(error**2)),grad_norm=float(np.linalg.norm(grad)),weight_norm=float(np.linalg.norm(self.w)),samples=len(ix))
 def agent(self,backend='native'):
  return Agent('old',self.policy,self.w.reshape(3,8),backend) if self.arm=='L' else TupleAgent(self.policy,self.base,self.w,backend)
 def save(self,path):
  d={k:(v.tolist() if isinstance(v,np.ndarray) else copy.deepcopy(v)) for k,v in self.__dict__.items() if k!='rng'};d['rng']=self.rng.bit_generator.state;d['format']='tuple-learner-state-v1';Path(path).write_text(json.dumps(d)+'\n')
 @classmethod
 def load(cls,path):
  d=json.loads(Path(path).read_text());assert d.pop('format')=='tuple-learner-state-v1';obj=cls.__new__(cls);rng=d.pop('rng');obj.__dict__.update(d)
  for k in ('policy','base','w','m','v','visits'):setattr(obj,k,np.asarray(getattr(obj,k),dtype=np.int64 if k=='visits' else np.float64))
  if obj.order is not None:obj.order=np.array(obj.order,dtype=np.int64)
  obj.rng=np.random.default_rng();obj.rng.bit_generator.state=rng;return obj
