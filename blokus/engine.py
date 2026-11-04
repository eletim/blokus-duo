"""Exact rules; coordinates are (row, column), zero based."""
from dataclasses import dataclass, field
N = 14
START = ((4, 4), (9, 9))
ORTH = ((1,0),(-1,0),(0,1),(0,-1))
DIAG = ((1,1),(1,-1),(-1,1),(-1,-1))
BOARD_MASK = (1 << (N*N)) - 1
FIRST_COL = sum(1 << (r*N) for r in range(N))
LAST_COL = FIRST_COL << (N-1)
def norm(cells):
    a,b = min(x for x,y in cells), min(y for x,y in cells)
    return tuple(sorted((x-a,y-b) for x,y in cells))
def orientations(cells):
    out=set()
    for flip in (1,-1):
        q=[(x,flip*y) for x,y in cells]
        for _ in range(4):
            out.add(norm(q)); q=[(y,-x) for x,y in q]
    return tuple(sorted(out))
def pieces():
    level={((0,0),)}; out=[]
    for size in range(1,6):
        out.extend(sorted(level))
        nxt=set()
        for p in level:
            for x,y in p:
                for dx,dy in ORTH:
                    if (x+dx,y+dy) not in p:
                        nxt.add(min(orientations(p+((x+dx,y+dy),))))
        level=nxt
    return tuple(out)
PIECES=pieces(); ORIENTS=tuple(orientations(p) for p in PIECES)
SIZES=tuple(map(len,PIECES))
def bit(c): return 1 << (c[0]*N+c[1])
def mask(cells): return sum(bit(c) for c in cells)
def neighbors(cells, dirs):
    return {(x+dx,y+dy) for x,y in cells for dx,dy in dirs if 0<=x+dx<N and 0<=y+dy<N}
def cells_of(m):
    while m:
        b=m & -m; k=b.bit_length()-1; yield divmod(k,N); m-=b
@dataclass(frozen=True)
class Move:
    piece:int
    orientation:int
    row:int
    col:int
    @property
    def cells(self):
        return tuple((x+self.row,y+self.col) for x,y in ORIENTS[self.piece][self.orientation])
@dataclass(frozen=True)
class Placement:
    move:Move
    bits:int
    edges:int
    corners:int
PLACEMENTS=[]; BY_POINT=[[] for _ in range(N*N)]
for p,os in enumerate(ORIENTS):
    for o,shape in enumerate(os):
        for r in range(N-max(x for x,y in shape)):
            for c in range(N-max(y for x,y in shape)):
                m=Move(p,o,r,c); cs=m.cells
                q=Placement(m,mask(cs),mask(neighbors(cs,ORTH)),mask(neighbors(cs,DIAG)))
                idx=len(PLACEMENTS); PLACEMENTS.append(q)
                for x,y in cs: BY_POINT[x*N+y].append(idx)
@dataclass
class State:
    boards:list=field(default_factory=lambda:[0,0])
    remaining:list=field(default_factory=lambda:[(1<<21)-1]*2)
    last:list=field(default_factory=lambda:[None,None])
    turn:int=0
    passes:int=0
    @property
    def done(self): return self.passes>=2
    def score(self,p):
        return (20 if self.last[p]==0 else 15) if not self.remaining[p] else -sum(SIZES[i] for i in range(21) if self.remaining[p]>>i&1)
    def render(self):
        return '   '+''.join(str(i%10) for i in range(N))+'\n'+'\n'.join(f'{r:2} '+''.join('X' if self.boards[0]&bit((r,c)) else 'O' if self.boards[1]&bit((r,c)) else '*' if (r,c) in START else '.' for c in range(N)) for r in range(N))
def frontier(s,p):
    own=s.boards[p]; occupied=s.boards[0]|s.boards[1]
    # Mask before horizontal shifts to prevent wrapping across row boundaries.
    horizontal=((own & ~FIRST_COL) >> 1) | ((own & ~LAST_COL) << 1)
    edge=(horizontal | (own >> N) | (own << N)) & BOARD_MASK
    f=(((horizontal >> N) | (horizontal << N)) & BOARD_MASK) if own else bit(START[p])
    return f & ~occupied & ~edge, edge

def legal_moves(s):
    if s.done: return []
    p=s.turn; f,e=frontier(s,p); forbidden=s.boards[0]|s.boards[1]|e
    ids=set()
    for r,c in cells_of(f): ids.update(BY_POINT[r*N+c])
    return [PLACEMENTS[i] for i in sorted(ids) if s.remaining[p]>>PLACEMENTS[i].move.piece&1 and not PLACEMENTS[i].bits&forbidden]

def is_legal(s,m):
    """Independent coordinate checker: does not call legal_moves/frontier."""
    if s.done: return False
    if m is None:
        # Exhaustive independent pass verification.
        return not any(is_legal(s,q.move) for q in PLACEMENTS)
    if not isinstance(m,Move) or not 0<=m.piece<21 or not 0<=m.orientation<len(ORIENTS[m.piece]): return False
    if not s.remaining[s.turn]>>m.piece&1: return False
    cs=set(m.cells)
    if any(not (0<=x<N and 0<=y<N) for x,y in cs): return False
    own=set(cells_of(s.boards[s.turn])); other=set(cells_of(s.boards[1-s.turn]))
    if cs & (own|other): return False
    if any((x+dx,y+dy) in own for x,y in cs for dx,dy in ORTH): return False
    return START[s.turn] in cs if not own else any((x+dx,y+dy) in own for x,y in cs for dx,dy in DIAG)

def play(s,m,validate=True):
    if validate and not is_legal(s,m): raise ValueError('Illegal move/pass')
    if s.done: raise ValueError('Game over')
    p=s.turn
    if m is None: s.passes+=1
    else:
        s.boards[p]|=mask(m.cells); s.remaining[p]&=~(1<<m.piece); s.last[p]=m.piece; s.passes=0
    s.turn=1-p
