"""Non-neural candidate features, terminal-outcome linear value and bounded search."""
import copy,json,random,time,hashlib
from functools import lru_cache
from pathlib import Path
import numpy as np
from .engine import State,PLACEMENTS,BOARD_MASK,FIRST_COL,LAST_COL,frontier,cells_of,play,legal_moves,SIZES
from .linear_features import ACTIVE,PRIOR,probabilities,inputs as old_inputs
FORMAT='linear-outcome-v2'
def adjacent(bits):
 return (((bits&~FIRST_COL)>>1)|((bits&~LAST_COL)<<1)|(bits>>14)|(bits<<14))&BOARD_MASK

def component_sizes(allowed):
 out=np.zeros(197,dtype=np.float64)
 while allowed:
  component=allowed&-allowed
  while True:
   nxt=component|(adjacent(component)&allowed)
   if nxt==component:break
   component=nxt
  n=component.bit_count()/196
  for r,c in cells_of(component):out[r*14+c]=n
  allowed&=~component
 return out

@lru_cache(maxsize=1)
def geometry_tables():
 cells=np.full((len(PLACEMENTS),5),196,np.int64);corners=np.full((len(PLACEMENTS),20),196,np.int64);pieces=[];mapping={}
 for i,q in enumerate(PLACEMENTS):
  a=[r*14+c for r,c in cells_of(q.bits)];b=[r*14+c for r,c in cells_of(q.corners&~(q.bits|q.edges))]
  cells[i,:len(a)]=a;corners[i,:len(b)]=b;pieces.append(q.move.piece);mapping[q.move]=i
 for a in (cells,corners):a.flags.writeable=False
 return cells,corners,np.array(pieces),mapping

def enriched_inputs(s,backend='native'):
 if backend=='native':
  from .ppo_native import scan
  ids,raw=scan(s);qs=[PLACEMENTS[i] for i in ids];x=np.asarray(raw[:,ACTIVE],dtype=np.float64)
 else:
  x,qs,_=old_inputs(s,'python');ids=np.array([geometry_tables()[3][q.move] for q in qs],dtype=np.int64)
 phase=0 if s.remaining[s.turn].bit_count()>14 else 1 if s.remaining[s.turn].bit_count()>7 else 2
 if not qs:return np.empty((0,13)),qs,phase
 cells,corners,pieces,_=geometry_tables();ps=pieces[ids];counts=np.bincount(ps,minlength=21)
 occupied=s.boards[0]|s.boards[1];f,e=frontier(s,s.turn);of,_=frontier(s,1-s.turn)
 components=component_sizes(BOARD_MASK&~(occupied|e));reach=components[corners[ids]].max(1)
 weights=np.zeros(197)
 for r,c in cells_of(of):
  point=1<<(r*14+c);weights[r*14+c]=(1+(adjacent(point)&~occupied).bit_count())/5
 block=weights[cells[ids]].sum(1)/5
 scarcity=-np.log1p(counts[ps])/np.log(201)
 return np.column_stack((x,scarcity,reach,block)),qs,phase

def outcome(s,player):
 if not s.done:raise ValueError('outcome requires terminal')
 d=s.score(player)-s.score(1-player);return float((d>0)-(d<0))

def value_features(s,backend='native'):
 # Antisymmetric player differences; phase is invariant to swapping player labels.
 stats=[];occ=s.boards[0]|s.boards[1]
 for player in (s.turn,1-s.turn):
  t=State(s.boards.copy(),s.remaining.copy(),s.last.copy(),player,s.passes)
  if backend=='native':
   from .ppo_native import scan
   ids,_=scan(t);pieces={PLACEMENTS[i].move.piece for i in ids};n=len(ids)
  else:
   qs=legal_moves(t);pieces={q.move.piece for q in qs};n=len(qs)
  f,e=frontier(s,player);components=component_sizes(BOARD_MASK&~(occ|e));reach=max((components[r*14+c] for r,c in cells_of(f)),default=0)
  stats.append(np.array([s.score(player)/89,-s.remaining[player].bit_count()/21,f.bit_count()/40,np.log1p(n)/8,len(pieces)/21,sum(SIZES[i]==5 for i in pieces)/12,reach,float(s.remaining[player]&1)]))
 mean=(s.remaining[0].bit_count()+s.remaining[1].bit_count())/2;phase=0 if mean>14 else 1 if mean>7 else 2
 phi=np.zeros((3,8));phi[phase]=stats[0]-stats[1];return phi

class Agent:
 def __init__(self,mode='new',weights=None,value=None,backend='native'):
  if mode not in ('old','new'):raise ValueError('mode old/new')
  if backend not in ('python','native'):raise ValueError('backend must be python or native')
  self.mode=mode;self.backend=backend;self.weights=np.tile(np.r_[PRIOR,np.zeros(3)] if mode=='new' else PRIOR,(3,1)) if weights is None else np.array(weights,dtype=float)
  self.value_weights=np.zeros((3,8)) if value is None else np.array(value,dtype=float)
  if self.weights.shape!=(3,13 if mode=='new' else 10) or self.value_weights.shape!=(3,8) or not np.isfinite(self.weights).all() or not np.isfinite(self.value_weights).all():raise ValueError('invalid linear model')
 def inputs(self,s):return enriched_inputs(s,self.backend) if self.mode=='new' else old_inputs(s,self.backend)
 def act(self,s,rng):
  x,qs,p=self.inputs(s)
  if not qs:return None
  score=x@self.weights[p];return qs[int(rng.choice(np.flatnonzero(score>=score.max()-1e-7)))].move
 def sample(self,s,rng):
  x,qs,phase=self.inputs(s);g=np.zeros_like(self.weights)
  if not qs:return None,g,0.
  p=probabilities(x,self.weights[phase]);i=int(rng.choice(len(qs),p=p));g[phase]=x[i]-p@x
  return qs[i].move,g,float(-(p*np.log(p.clip(1e-300))).sum())
 def value(self,s):return outcome(s,s.turn) if s.done else float(np.clip(np.sum(self.value_weights*value_features(s,self.backend)),-1,1))
 def save(self,path,metadata=None):Path(path).write_text(json.dumps(dict(format=FORMAT,mode=self.mode,weights=self.weights.tolist(),value=self.value_weights.tolist(),metadata=metadata or {}),indent=2)+'\n')
 @classmethod
 def load(cls,path,backend='native'):
  x=json.loads(Path(path).read_text())
  if x.get('format')!=FORMAT:raise ValueError('unknown non-neural model format')
  return cls(x['mode'],x['weights'],x['value'],backend)


 @classmethod
 def load_pair(cls,policy_path,value_path,backend='native',policy_sha256=None,value_sha256=None):
  # Both inputs must be the explicit v2 schema (same phase/value semantics).
  paths=[Path(policy_path),Path(value_path)];hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
  for actual,expected in zip(hashes,(policy_sha256,value_sha256)):
   if expected is not None and actual!=expected:raise ValueError('linear source SHA256 mismatch')
  policy=cls.load(paths[0],backend);value=cls.load(paths[1],backend)
  agent=cls(policy.mode,policy.weights,value.value_weights,backend)
  agent.provenance=dict(policy_sha256=hashes[0],value_sha256=hashes[1],value_schema='linear-outcome-v2/3x8/player-antisymmetric-mean-piece-phase')
  return agent

class Search:
 def __init__(self,agent,ms=6,width=6,replies=3,depth=2,learned=True):
  if not np.isfinite(ms) or ms<0 or not isinstance(depth,int) or depth<1:raise ValueError('invalid search budget/depth')
  if any(w is not None and (not isinstance(w,int) or w<1) for w in (width,replies)):raise ValueError('width/replies must be positive or None')
  self.agent=agent;self.ms=ms;self.width=width;self.replies=replies;self.depth=depth;self.learned=learned;self.stats=[]
 def leaf(self,s):
  if s.done:return outcome(s,s.turn)
  if self.learned:return self.agent.value(s)
  # Fixed nonlearned WDL-scale proxy, not true game value.
  return (s.score(s.turn)-s.score(1-s.turn))/109
 def act(self,s,rng):
  begin=time.perf_counter();deadline=begin+self.ms/1000 if self.ms else float('inf');nodes=0
  x,qs,p=self.agent.inputs(s)
  if not qs:self.stats.append(dict(ms=(time.perf_counter()-begin)*1000,nodes=0,fallback=False,forced_pass=True,complete=True));return None
  scores=x@self.agent.weights[p];baseline=qs[int(rng.choice(np.flatnonzero(scores>=scores.max()-1e-7)))].move
  order=np.argsort(-scores,kind='stable');roots=[qs[i].move for i in order[:self.width]] if self.width else [qs[i].move for i in order]
  def visit(t,depth):
   nonlocal nodes
   nodes+=1
   if time.perf_counter()>deadline-.0002:raise TimeoutError
   if t.done:return outcome(t,t.turn)
   if depth==0:return self.leaf(t)
   a,c,k=self.agent.inputs(t)
   if c:
    ids=np.argsort(-(a@self.agent.weights[k]),kind='stable');moves=[c[i].move for i in (ids[:self.replies] if self.replies else ids)]
   else:moves=[None]
   best=-float('inf')
   for m in moves:
    child=copy.deepcopy(t);play(child,m,validate=False);best=max(best,-visit(child,depth-1))
   return best
  values=[];fallback=False
  try:
   for m in roots:
    child=copy.deepcopy(s);play(child,m,validate=False);values.append(-visit(child,self.depth-1))
   if time.perf_counter()>deadline:raise TimeoutError
   choice=int(rng.choice(np.flatnonzero(np.asarray(values)>=max(values)-1e-12)));move=roots[choice]
  except TimeoutError:move=baseline;fallback=True
  self.stats.append(dict(ms=(time.perf_counter()-begin)*1000,nodes=nodes,fallback=fallback,complete=not fallback,root_candidates=len(roots),completed_roots=len(values),legal_candidates=len(qs),overshoot=time.perf_counter()>deadline,values=values if not fallback else []))
  return move
