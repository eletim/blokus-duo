import numpy as np
from collections import Counter
from blokus.scaling_data import strata

def test_nested_game_splits_are_balanced():
 for n in [360,1440,3600]:
  folds=[{g for g in range(n) if strata(g)['fold']!=f} for f in range(3)]
  for f,ids in enumerate(folds):
   assert len(ids)==n*2//3
   counts=Counter((strata(g)['pool'],strata(g)['seat']) for g in ids)
   assert len(counts)==6 and len(set(counts.values()))==1
   if n>360:assert {g for g in range(360) if strata(g)['fold']!=f}<=ids
 test=[{100000+g for g in range(360) if strata(g)['fold']==f} for f in range(3)]
 assert all(len(v)==120 for v in test) and not any(test[i]&test[j] for i in range(3) for j in range(i))
 assert not set(range(3600))&set.union(*test)

def test_per_game_equal_weight_and_unknown_mask():
 import torch
 from blokus.scaling_train import per_game,summary
 class Zero(torch.nn.Module):
  def forward(self,b,s,x):return torch.zeros(x.shape[:2])
 data=dict(board=torch.zeros(3,6,14,14),state=torch.zeros(3,49),x=torch.zeros(3,2,39),mask=torch.tensor([[True,False]]*3),target=torch.tensor([[2.,float('nan')],[2.,float('nan')],[4.,float('nan')]]),initial=torch.zeros(3,2),game=torch.tensor([1,1,2]))
 values=per_game(Zero(),data,torch.arange(3));assert len(values)==2 and summary(values)['mse']==10
 assert summary(values)['regret']==0 and summary(values)['zero_regret']==0

def test_failed_or_incomplete_learning_gate_blocks_battles(tmp_path,monkeypatch):
 import json,pytest
 from blokus import scaling_battle
 monkeypatch.setattr(scaling_battle,'ROOT',tmp_path)
 for advance,final in [(False,True),(True,False)]:
  (tmp_path/'decision.json').write_text(json.dumps(dict(advance=advance,final=final)))
  with pytest.raises(AssertionError,match='gate failed or curve incomplete'):scaling_battle.run()
