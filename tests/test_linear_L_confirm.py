import numpy as np
from scripts import linear_L_confirm as e

def test_two_comparison_interval_and_seat_groups():
 games=[dict(prefix_seed=p,seat=seat,point=float(p%2),difference=2*(p%2)-1) for p in range(8) for seed in range(3) for seat in (0,1)]
 d=e.interval(games,6)
 assert d['games']==48 and d['point']==.5 and d['difference']==0
 assert d['point_ci_bonferroni2'][0]<=d['point_ci95'][0]<=d['point']<=d['point_ci95'][1]<=d['point_ci_bonferroni2'][1]
 assert e.interval([x for x in games if x['seat']==0],3)['point']==.5

def test_frozen_L_policy_and_value_source():
 from blokus.linear_value import Agent
 for s in e.SEEDS:
  a=Agent.load(e.source('L',s));f=Agent.load(e.source('F',s))
  assert a.mode=='old' and np.array_equal(a.weights,f.weights)
  assert not np.array_equal(a.value_weights,f.value_weights)
