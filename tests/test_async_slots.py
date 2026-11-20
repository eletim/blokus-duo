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

def test_resume_rejects_changed_configuration_before_cuda(tmp_path,monkeypatch):
 import json
 from scripts.async_generation import bench
 monkeypatch.setattr('os.sched_setaffinity',lambda *args:None)
 bank=tmp_path/'bank.json';bank.write_text(json.dumps({'games':[],'models':{}}))
 out=tmp_path/'out';out.mkdir();(out/'run.json').write_text('{}')
 marker=out/'completed.json.gz';marker.write_bytes(b'preserved')
 import pytest
 with pytest.raises(AssertionError,match='resume configuration mismatch'):
  bench(bank,out,concurrency=512,graphs=True,fast=True)
 assert marker.read_bytes()==b'preserved'

def test_duplicate_game_or_rng_is_rejected_before_cuda(tmp_path,monkeypatch):
 import json,pytest
 from scripts.async_generation import bench
 monkeypatch.setattr('os.sched_setaffinity',lambda *args:None)
 for rows in ([{'seed':1,'id':1,'rng_seed':1}]*2,
              [{'seed':1,'id':1,'rng_seed':1},{'seed':1,'id':2,'rng_seed':1}]):
  bank=tmp_path/'bank.json';bank.write_text(json.dumps({'games':rows,'models':{}}))
  with pytest.raises(AssertionError):bench(bank,tmp_path/'out')
