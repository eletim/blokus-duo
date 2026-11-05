"""Deterministic node-limited diagnostics. No TT and no wall-clock cutoff."""
import numpy as np
from .engine import legal_moves,play
from .research import clone
from .ai import features,heuristic
from .value_search import SCALE,exact_terminal,score_value

class NodeLimit(Exception):pass

def ordered(s):
    qs=legal_moves(s)
    if not qs:return [None]
    x,_=features(s,qs)
    return [qs[int(i)].move for i in np.argsort(-heuristic(x),kind='stable')]

def child(s,m):
    c=clone(s);play(c,m,validate=False);return c

def row(m):return [m.piece,m.orientation,m.row,m.col] if m else None

class NodeSearch:
    def __init__(self,evaluator,limit=2048,width=6):
        self.evaluator=evaluator;self.limit=limit;self.width=width;self.nodes=0
    def consume(self,n=1):
        if self.nodes+n>self.limit:raise NodeLimit()
        self.nodes+=n
    def value(self,s,depth,full=False):
        self.consume()
        if exact_terminal(s):return score_value(s)*SCALE
        if depth<=0 and not full:return float(self.evaluator.batch([s])[0])*SCALE
        moves=ordered(s);moves=moves if full else moves[:self.width]
        children=[child(s,m) for m in moves]
        if depth==1 and not full:
            self.consume(len(children))
            return float((-self.evaluator.batch(children)*SCALE).max())
        return max(-self.value(c,depth-1,full) for c in children)
    def roots(self,s,root_width,reply_width,depth,full=False):
        rows=[];interrupted=False
        for m in ordered(s)[:root_width]:
            try:
                self.consume();c=child(s,m)
                if exact_terminal(c):v=-score_value(c)*SCALE
                else:
                    replies=ordered(c);replies=replies if full else replies[:reply_width]
                    v=min(self.value(child(c,r),depth-2,full) for r in replies)
                rows.append(dict(move=row(m),value=v))
            except NodeLimit:interrupted=True;break
        return dict(values=rows,nodes=self.nodes,interrupted=interrupted,chosen=max(rows,key=lambda r:r['value'])['move'] if rows else row(ordered(s)[0]))
