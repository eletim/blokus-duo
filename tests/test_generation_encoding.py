import copy,random
import numpy as np
from blokus.engine import State,play,legal_moves
from blokus.value_search import encode as reference
from blokus.generation_encoding import encode,child,observe
from blokus.ppo_native import observe as native

def test_encoding_and_copy_exact():
 rng=random.Random(9812)
 for game in range(8):
  s=State()
  while not s.done:
   for ch in (4,6):
    b,x=reference(s);bb,xx=encode(s,ch)
    assert np.array_equal(b[:ch],bb) and np.array_equal(x,xx)
    obs,qs=observe(s,ch);ref,rqs=native(s,ch)
    assert [q.move for q in qs]==[q.move for q in rqs]
    assert all(np.array_equal(a,b) for a,b in zip(obs,ref))
   moves=legal_moves(s);m=rng.choice(moves).move if moves else None
   before=copy.deepcopy(s);c=child(s,m);assert s==before
   play(s,m,validate=False);assert c==s
  assert np.array_equal(encode(s)[1],reference(s)[1])
