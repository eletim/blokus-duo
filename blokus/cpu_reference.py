"""Frozen bd962d7 CPU oracle; intentionally unoptimized, never production default."""
import time
import numpy as np
from .engine import *
from .value_search import encode
DIM=12
EXTRA=27
TIMES={"features":0.}

def legal_moves(s):
    if s.done: return []
    p=s.turn; f,e=frontier(s,p); forbidden=s.boards[0]|s.boards[1]|e
    ids=set()
    for r,c in cells_of(f): ids.update(BY_POINT[r*N+c])
    return [PLACEMENTS[i] for i in sorted(ids) if s.remaining[p]>>PLACEMENTS[i].move.piece&1 and not PLACEMENTS[i].bits&forbidden]

def features(s,qs):
    t=time.perf_counter(); p=s.turn; own=s.boards[p]; occ=own|s.boards[1-p]
    f,e=frontier(s,p); of,oe=frontier(s,1-p)
    rows=[]
    for q in qs:
        cs=q.move.cells; size=SIZES[q.move.piece]
        nf=(f|q.corners)&~(occ|q.bits|e|q.edges)
        freed=nf.bit_count()-f.bit_count()
        center=sum(13-abs(x-6.5)-abs(y-6.5) for x,y in cs)/size/13
        reach=sum(abs(x-START[p][0])+abs(y-START[p][1]) for x,y in cs)/size/26
        rows.append([size/5,freed/10,(q.bits&of).bit_count()/5,center,reach,
                     (q.edges&of).bit_count()/10, nf.bit_count()/40,
                     (q.corners&~occ&~q.bits).bit_count()/20,
                     float(size==1),s.remaining[p].bit_count()/21,
                     (s.boards[p].bit_count()-s.boards[1-p].bit_count())/89,
                     (q.edges&s.boards[1-p]).bit_count()/10])
    st=np.array([float(s.remaining[a]>>i&1) for a in (p,1-p) for i in range(21)]+[own.bit_count()/89,s.boards[1-p].bit_count()/89,f.bit_count()/40,of.bit_count()/40],dtype=np.float32)
    x=np.asarray(rows,dtype=np.float32).reshape(-1,DIM)
    TIMES['features']+=time.perf_counter()-t
    return x,st

def inputs(s,qs,x=None,st=None):
    if x is None:x,st=features(s,qs)
    extra=np.zeros((len(qs),EXTRA),np.float32)
    for i,q in enumerate(qs):
        cells=q.move.cells;rs=[r for r,c in cells];cs=[c for r,c in cells]
        extra[i,q.move.piece]=1
        extra[i,21:]=[sum(rs)/len(rs)/13,sum(cs)/len(cs)/13,min(rs)/13,min(cs)/13,(max(rs)-min(rs))/4,(max(cs)-min(cs))/4]
    state=np.r_[st,np.array([s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],np.float32)]
    return np.concatenate((x,extra),axis=1),state.astype(np.float32)

def observe(s,qs=None):
 if s.done:raise ValueError('terminal has no policy actions')
 qs=legal_moves(s) if qs is None else qs;b,st=encode(s)
 if qs:
  x,_=inputs(s,qs);cells=np.full((len(qs),5),196,np.int64)
  for i,q in enumerate(qs):
   for j,(y,col) in enumerate(q.move.cells):cells[i,j]=14*y+col
 else:x=np.zeros((1,39),np.float32);cells=np.full((1,5),196,np.int64)
 return (b,st,x,cells),qs
