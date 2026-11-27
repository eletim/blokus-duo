"""Independent experimental raw-point regressor. No inference dependencies."""
import json
from pathlib import Path
from .engine import frontier,cells_of
FORMAT='margin-ridge-v1'


def features(s):
    """Player-relative differences, gated by symmetric material phase."""
    a,b=s.turn,1-s.turn
    x=[float((s.remaining[a]>>i&1)-(s.remaining[b]>>i&1)) for i in range(21)]
    for bits in (s.boards,[frontier(s,p)[0] for p in (0,1)]):
        bins=[0.0]*16
        for player,sign in ((a,1),(b,-1)):
            for r,c in cells_of(bits[player]):bins[min(r//4,3)*4+min(c//4,3)]+=sign/16
        x+=bins
    x += [(s.score(a)-s.score(b))/89,
          (frontier(s,a)[0].bit_count()-frontier(s,b)[0].bit_count())/40,
          (s.remaining[a].bit_count()-s.remaining[b].bit_count())/21,
          1.0 if a==0 else -1.0]
    count=(s.remaining[0].bit_count()+s.remaining[1].bit_count())/2
    phase=0 if count>14 else 1 if count>7 else 2
    return [v if phase==k else 0.0 for k in range(3) for v in x]


class MarginValue:
    def __init__(self,path,komi=0):
        self.model=json.loads(Path(path).read_text())
        if self.model['format']!=FORMAT:raise ValueError('unknown point model')
        self.weights=self.model['weights'];self.komi=float(komi)
    def raw(self,s):
        if s.done:return float(s.score(s.turn)-s.score(1-s.turn))
        return max(-109.0,min(109.0,sum(w*x for w,x in zip(self.weights,features(s)))))
    def __call__(self,s):
        # k points to second: first-player margin -= k; second margin += k.
        return self.raw(s)+(-self.komi if s.turn==0 else self.komi)
