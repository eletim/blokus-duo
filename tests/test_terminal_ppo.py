import numpy as np
import pytest
import torch
from blokus.terminal_ppo import TerminalNet,observe,collate,terminal_return
from blokus.engine import State,play,legal_moves

def test_all_legal_mask_and_pass():
 s=State();obs,qs=observe(s);assert len(qs)==414
 passed=State(remaining=[0,1]);o,q=observe(passed);assert not q and len(o[2])==1
 m=TerminalNet();logits,v=m(*collate([obs,o],'cpu'));assert torch.isfinite(logits[0]).all();assert torch.isneginf(logits[1,1:]).all();assert logits[1,0]==0
 p=torch.distributions.Categorical(logits=logits).probs;assert p[1,0]==1 and p[1,1:].sum()==0

def test_terminal_bonuses_returns_and_pass_perspective():
 s=State(remaining=[0,0],last=[0,1]);assert s.score(0)==20 and s.score(1)==15
 with pytest.raises(ValueError):terminal_return(s,0)
 play(s,None);assert s.turn==1 and not s.done
 play(s,None);assert s.done and terminal_return(s,0)==1 and terminal_return(s,1)==-1
 s.last=[1,1];assert terminal_return(s,0)==terminal_return(s,1)==0
 # Repeated passes by one side, moves by the other: return belongs to player, not trace parity.
 s=State(remaining=[0,1]);play(s,None);q=legal_moves(s)[0];play(s,q.move);assert s.turn==0 and s.passes==0
 play(s,None);play(s,None);assert s.done;assert terminal_return(s,0)==-terminal_return(s,1)

def test_exact_placement_input_gradient():
 import json
 from blokus.engine import Move
 r=json.load(open('artifacts/candidate-mask/collision.json'));s=State(**r['state']);by={q.move:q for q in legal_moves(s)};qs=[by[Move(*m)] for m in r['moves']];o,_=observe(s,qs);assert np.array_equal(o[2][0],o[2][1]);assert not np.array_equal(o[3][0],o[3][1])
 torch.manual_seed(19);model=TerminalNet();torch.nn.init.normal_(model.policy[-1].weight);args=collate([o],'cpu');l,_=model(*args);assert not torch.isclose(l[0,0],l[0,1]);l.sum().backward();assert model.position.weight.grad.abs().sum()>0

def test_initial_logits_match_heuristic():
 from blokus.ai import heuristic
 o,_=observe(State());l,v=TerminalNet()(*collate([o],'cpu'));np.testing.assert_allclose(l.detach()[0].numpy(),heuristic(o[2][:,:12]),atol=1e-5);assert v==0

def test_holdout_rejects_seen_prefix_and_state_before_evaluation():
 from blokus.terminal_eval import disjoint_fixture
 from blokus.tournament import make_prefixes,prefix_state
 from blokus.visited_data import state_hash
 initial=make_prefixes(1500000,16);blocked={initial['prefixes'][0]['id']};states={state_hash(prefix_state(initial['prefixes'][1]))}
 a=disjoint_fixture(1500000,4,blocked,states);b=disjoint_fixture(1500000,4,blocked,states)
 assert a==b and len(a['prefixes'])==4
 assert not blocked & {p['id'] for p in a['prefixes']}
 assert not states & {state_hash(prefix_state(p)) for p in a['prefixes']}
