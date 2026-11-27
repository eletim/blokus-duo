import numpy as np
from scripts.komi4_scale import task,pack_training

def test_pairs_share_prefix_and_splits_do_not_share_seed():
 for i in range(30):
  a=task('comparison',i,opp='ridge',seat=0);b=task('comparison',i,opp='gen4',seat=1)
  assert a['prefix_seed']==b['prefix_seed'] and a['group']==b['group']
  assert task('train',i,1)['prefix_seed']!=task('test',i)['prefix_seed']
  assert task('train',i,1)['prefix_seed']!=task('train',i,2)['prefix_seed']

def test_each_epoch_covers_old_and_new_with_equal_total_mass():
 def data(n):return {**{k:np.zeros((n,1)) for k in ['b','st','ex']},'y':np.zeros(n),'w':np.ones(n)}
 merged=pack_training(data(4),[data(8),data(8)])
 assert len(merged['y'])==20
 assert np.isclose(merged['w'][:16].sum(),.5)
 assert np.isclose(merged['w'][16:].sum(),.5)

def test_live_scale_replay_has_real_komi_terminal_and_missing_root_prediction():
 from blokus.console import Sources,scale_games
 if not scale_games('train-g1'):return
 r=Sources('/unused','/unused').scale_game('train-g1',0)
 assert r['komi']==4
 assert r['predictions']['values'][r['prefix_plies']] is None
 end=r['predictions']['values'][-1];s=r['frames'][-1]['scores']
 assert end['adjusted']==(s[0]-s[1]-4)*(1 if end['player']==0 else -1)
