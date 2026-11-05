from collections import Counter
from blokus import terminal_ppo as p

def test_pool_cycle_counts_and_seats():
 counts=Counter();seats=Counter()
 for iteration in range(6144,6150):
  a=p.pool_modes(iteration,16,'A');b=p.pool_modes(iteration,16,'B')
  assert Counter(a)==dict(current=8,past=4,heuristic=4)
  for i,m in enumerate(b):counts[m]+=1;seats[m,(i//4)%2]+=1
 assert counts==dict(current=48,heuristic=24,past=8,gen2=8,gen2048=8)
 for m in counts:assert seats[m,0]==seats[m,1]

def test_frozen_opponents_never_supply_training_targets(monkeypatch):
 used=[]
 def sample(model,obs):
  used.append(model);return [0]*len(obs),[0.]*len(obs),[0.]*len(obs)
 monkeypatch.setattr(p,'sample_batch',sample)
 rec,games,_=p.collect('learner','past',6144,1401,16,fixed_pool={'gen2':'gen2','gen2048':'gen2048'})
 assert {'learner','past','gen2','gen2048'}<=set(used)
 assert len(rec)==sum(len(g['targets']) for g in games)
 for g in games:
  if g['opponent']!='current':assert all(t['player']==g['learner_seat'] for t in g['targets'])
 assert Counter(r['opponent'] for r in rec)==Counter({m:sum(len(g['targets']) for g in games if g['opponent']==m) for m in {g['opponent'] for g in games}})

def test_pool_lock_rejects_changed_frozen_model(tmp_path,monkeypatch):
 import json,pytest
 from blokus import ppo_pool_eval as ev
 from blokus.research import sha
 model=tmp_path/'model.pt';model.write_bytes(b'fixed')
 teacher=tmp_path/'teacher.pt';teacher.write_bytes(b'teacher')
 (tmp_path/'lock.json').write_text(json.dumps(dict(gen2={},resume={},previous={},fixed={'1401':{'gen2':dict(path=str(model),sha256=sha(model))}},teacher=dict(sha256=sha(teacher)))))
 monkeypatch.setattr(ev,'R',tmp_path);monkeypatch.setattr(ev.e,'BASE',teacher)
 ev.lock();model.write_bytes(b'changed')
 with pytest.raises(AssertionError):ev.lock()
