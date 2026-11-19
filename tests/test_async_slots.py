import multiprocessing as mp
import numpy as np
from scripts.async_generation import Slots,CAP

def test_shared_slots_are_isolated_and_typed():
 slots=Slots(mp.get_context('spawn'),2);a=slots.arrays();b=slots.arrays()
 for x,y in zip(a,b):
  x[0]=7
  assert np.all(y[0]==7) and np.all(y[1]==0)
 assert a[2].shape==(2,CAP,39) and a[3].dtype==np.int32
 # Output writes cannot corrupt observation buffers.
 a[4][0]=31;a[5][0]=.125
 assert np.all(a[0][0]==7) and a[5][1]==0
