"""Opt-in generation-only encoding; same float32 features, no policy change."""
import numpy as np
from .engine import State,frontier,START
from .ppo_native import scan,tables
from .engine import PLACEMENTS

def bits(n,count):
 return np.unpackbits(np.frombuffer(int(n).to_bytes((count+7)//8,'little'),dtype=np.uint8),bitorder='little')[:count]
def encode(s,channels=4):
 p=s.turn;q=1-p;fp=frontier(s,p)[0];fq=frontier(s,q)[0]
 board=np.stack([bits(n,196).reshape(14,14) for n in (s.boards[p],s.boards[q],fp,fq)])
 if channels==6:
  start=np.zeros((2,14,14),np.uint8)
  for i,side in enumerate((p,q)):start[i,START[side][0],START[side][1]]=1
  board=np.concatenate((board,start))
 elif channels!=4:raise ValueError('channels')
 state=np.empty(49,np.float32);state[:21]=bits(s.remaining[p],21);state[21:42]=bits(s.remaining[q],21)
 state[42:]=[s.boards[p].bit_count()/89,s.boards[q].bit_count()/89,fp.bit_count()/40,fq.bit_count()/40,s.score(p)/89,s.score(q)/89,s.passes/2]
 return board,state

def observe(s,channels=4):
 ids,x=scan(s,True);board,state=encode(s,channels)
 qs=[PLACEMENTS[i] for i in ids]
 if len(ids):cells=tables()[3][ids].copy()
 else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
 return (board,state,x,cells),qs

def child(s,m):
 from .engine import play
 z=State(s.boards.copy(),s.remaining.copy(),s.last.copy(),s.turn,s.passes);play(z,m,validate=False);return z
