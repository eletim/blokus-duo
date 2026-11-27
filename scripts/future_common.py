import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,hashlib,random,copy,time,gzip
from pathlib import Path
import numpy as np
import torch
from blokus.engine import State,Move,play,legal_moves,cells_of
from blokus.gen4 import load,verify
from blokus.ppo_search_deeper import DeepSearch
from blokus.ppo_runtime import state_hash
from blokus.future_occupancy import transform
R=Path('artifacts/future-occupancy');SEEDS=[1751401,1751402,1751403]
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def raw(m):return [m.piece,m.orientation,m.row,m.col] if m else None
def state(ms):
 s=State()
 for m in ms:play(s,Move(*m) if m else None,validate=True)
 return s
def canon(s):
 b=np.zeros((2,14,14),np.uint8)
 for p in [0,1]:
  for row,col in cells_of(s.boards[p]):b[p,row,col]=1
 tail=repr((s.remaining,s.last,s.turn,s.passes)).encode()
 return min(hashlib.sha256(transform(b,k).tobytes()+tail).hexdigest() for k in range(8))
def prefix(rng,n):
 s=State();ms=[]
 for i in range(n):
  qs=legal_moves(s);m=rng.choice(qs).move if qs else None;play(s,m,validate=True);ms.append(raw(m))
 return ms
