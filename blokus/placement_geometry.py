"""Bounded, state-independent immutable geometry; no RNG and no mutable array views."""
from functools import lru_cache
from .engine import START

@lru_cache(maxsize=32768)
def geometry(move):
    cells=move.cells;size=len(cells)
    center=sum(13-abs(x-6.5)-abs(y-6.5) for x,y in cells)/size/13
    reach=tuple(sum(abs(x-START[p][0])+abs(y-START[p][1]) for x,y in cells)/size/26 for p in (0,1))
    rs=[r for r,c in cells];cs=[c for r,c in cells]
    extra=[0.]*27;extra[move.piece]=1.
    extra[21:]=[sum(rs)/size/13,sum(cs)/size/13,min(rs)/13,min(cs)/13,(max(rs)-min(rs))/4,(max(cs)-min(cs))/4]
    indices=tuple(14*r+c for r,c in cells)+(196,)*(5-size)
    return center,reach,tuple(extra),indices
