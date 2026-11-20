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

def test_bonus_pass_and_bit_coordinates():
 for turn in (0,1):
  for last in (0,1):
   s=State(remaining=[0,0],last=[last,last],turn=turn,passes=2)
   assert s.score(turn)==(20 if last==0 else 15)
   for channels in (4,6):
    b,x=reference(s);bb,xx=encode(s,channels);assert np.array_equal(b[:channels],bb) and np.array_equal(x,xx)
 for cell in range(196):
  s=State(boards=[1<<cell,0]);b,x=encode(s);assert b[0].sum()==1 and b[0].ravel()[cell]==1
