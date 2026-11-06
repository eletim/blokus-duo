import copy,random
import numpy as np
import pytest
import torch
from blokus import ppo_runtime as r,ppo_native as n
from blokus.ppo_policy import PPOAgent,observe,TerminalNet,collate
from blokus.engine import State,START,frontier,bit,legal_moves,is_legal,play


def equal_obs(a,b):
 for x,y in zip(a,b):assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

def test_four_channels_frontier_start_not_board_corner():
 for turn in [0,1]:
  s=State(turn=turn);o,qs=observe(s,channels=4);b=o[0];assert b.shape==(4,14,14) and b[:2].sum()==0
  for index,player in enumerate([turn,1-turn]):
   assert frontier(s,player)[0]==bit(START[player]);assert b[index+2].sum()==1 and b[index+2,START[player][0],START[player][1]]==1
  assert len(qs)==414 and all(START[turn] in q.move.cells for q in qs)
  play(s,qs[0].move);a,_=observe(s,channels=4);assert a[0][3].sum()==frontier(s,turn)[0].bit_count()

def test_native_order_pack_boundary_pass_and_all_orientations():
 if n.library() is None:pytest.skip('optional native not built')
 states=[State(remaining=[1<<i,1<<i],turn=p) for i in range(21) for p in [0,1]]
 states += [State(boards=[bit(c),0],turn=p) for c in [(0,0),(0,13),(13,0),(13,13)] for p in [0,1]]
 states += [State(remaining=[0,1]),State(remaining=[0,0],last=[0,1],passes=1)]
 s=State();rng=random.Random(1740000)
 while not s.done:
  states.append(copy.deepcopy(s));qs=legal_moves(s);play(s,rng.choice(qs).move if qs else None,validate=False)
 for s in states:
  for channels in [4,6]:
   a,qs=n.observe(s,channels,'native');b,old=observe(s,channels=channels);assert qs==old;equal_obs(a,b)
   for q in qs:assert is_legal(s,q.move)
   a[0].fill(0);a[2].fill(0);a[3].fill(0);equal_obs(n.observe(s,channels,'native')[0],b)

def test_python_fallback(monkeypatch):
 monkeypatch.setattr(n,'library',lambda:None);s=State();equal_obs(n.observe(s,4,'auto')[0],observe(s,channels=4)[0])
 with pytest.raises(RuntimeError):n.observe(s,4,'native')
 with pytest.raises(ValueError):observe(State(),channels=5)
 with pytest.raises(ValueError):n.observe(State(passes=2))

def test_formats_preserve_six_and_reject_silent_conversion(tmp_path):
 agent=PPOAgent(1401,'cpu',generation='gen2');assert agent.channels==6
 m=TerminalNet(4);ck=dict(format='terminal-ppo-inference-v2',channels=4,model=m.state_dict(),iteration=0,seed=1);path=tmp_path/'four.pt';torch.save(ck,path)
 assert PPOAgent(1401,'cpu',checkpoint=path).channels==4
 ck['format']='terminal-ppo-inference-v1';torch.save(ck,path)
 with pytest.raises(ValueError):PPOAgent(1401,'cpu',checkpoint=path)
 ck['format']='terminal-ppo-inference-v2';ck['channels']=6;torch.save(ck,path)
 with pytest.raises(ValueError):PPOAgent(1401,'cpu',checkpoint=path)

def test_returns_pass_bonus_and_packed_cpu_update():
 r.seed_all(1740000);s=State(remaining=[0,0],last=[0,1]);play(s,None);play(s,None)
 assert r.terminal_return(s,0)==1 and r.terminal_return(s,1)==-1
 with pytest.raises(ValueError):r.terminal_return(State(),0)
 model=TerminalNet(4);past=copy.deepcopy(model);rec,games,t=r.collect(model,past,0,1740000,8,backend='python');opt=torch.optim.Adam(model.parameters(),lr=.0003);before=copy.deepcopy(model.state_dict());u=r.update(model,opt,rec,packed=True)
 assert u['grad_norm']>0 and np.isfinite(u['loss']) and any(not torch.equal(before[k],model.state_dict()[k]) for k in before)
 for g in games:
  if g['opponent']!='current':assert all(x['player']==g['learner_seat'] for x in g['targets'])
  assert all(x['return']==np.sign(g['scores'][x['player']]-g['scores'][1-x['player']]) for x in g['targets'])

@pytest.mark.parametrize('device',['cpu','cuda'])
def test_public_inference_native_matches_python(device,tmp_path):
 if device=='cuda' and not torch.cuda.is_available():pytest.skip('no CUDA')
 if n.library() is None:pytest.skip('optional native not built')
 for channels in [4,6]:
  path=None
  if channels==4:
   path=tmp_path/'inference.pt';torch.save(dict(format='terminal-ppo-inference-v2',channels=4,model=TerminalNet(4).state_dict(),iteration=0,seed=1),path)
  a=PPOAgent(device=device,checkpoint=path,generation='gen2',backend='python');b=PPOAgent(device=device,checkpoint=path,generation='gen2',backend='native');s=State()
  for i in range(6):
   ma=a.act(s,random.Random(i));mb=b.act(s,random.Random(i));assert ma==mb;play(s,ma)

def test_thread_local_native_buffers():
 if n.library() is None:pytest.skip('optional native not built')
 from concurrent.futures import ThreadPoolExecutor
 states=[State(turn=i%2,remaining=[(1<<21)-1,1<<(i%21)]) for i in range(16)]
 def check(s):
  actual=n.observe(s,4,'native')[0];expected=observe(s,channels=4)[0];equal_obs(actual,expected);return True
 with ThreadPoolExecutor(2) as pool:assert all(pool.map(check,states))
