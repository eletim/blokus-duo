"""Pure inference input encoding; byte-equivalent to the research value encoder."""
import numpy as np
from .engine import START, cells_of, frontier

def encode(s):
    board=np.zeros((6,14,14),dtype=np.uint8)
    fs=[frontier(s,p)[0] for p in (s.turn,1-s.turn)]
    for i,p in enumerate((s.turn,1-s.turn)):
        for r,c in cells_of(s.boards[p]):board[i,r,c]=1
        for r,c in cells_of(fs[i]):board[i+2,r,c]=1
        board[i+4,START[p][0],START[p][1]]=1
    st=np.array([float(s.remaining[p]>>i&1) for p in (s.turn,1-s.turn) for i in range(21)]+
                [s.boards[s.turn].bit_count()/89,s.boards[1-s.turn].bit_count()/89,fs[0].bit_count()/40,fs[1].bit_count()/40,
                 s.score(s.turn)/89,s.score(1-s.turn)/89,s.passes/2],dtype=np.float32)
    return board,st
