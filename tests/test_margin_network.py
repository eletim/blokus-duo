import copy
import torch
import numpy as np
from blokus.engine import State,Move,play
from blokus.margin_network import MarginNet,NetworkEvaluator,terminal_margin,inputs


def test_terminal_bonus_komi_and_both_views(tmp_path):
    m=MarginNet();p=tmp_path/'net.pt';torch.save(dict(model=m.state_dict(),komi=4),p)
    evaluator=NetworkEvaluator(p);s=State();s.remaining=[0,0];s.last=[1,0];s.passes=2
    assert terminal_margin(s)==-9 and evaluator(s)==-9
    s.turn=1
    assert terminal_margin(s)==9 and evaluator(s)==9


def test_model_uses_point_residual_and_gradient(tmp_path):
    torch.set_num_threads(1);s=State();play(s,Move(0,0,4,4),validate=True)
    b,st,e=inputs(s);model=MarginNet()
    for param in model.parameters():torch.nn.init.zeros_(param)
    # With zero learned residual, the output equals current observed margin + komi.
    output=model(torch.tensor(b[None]),torch.tensor(st[None]),torch.tensor(e[None]))
    assert abs(output.item()-terminal_margin(s))<1e-5
    loss=(output-20).square().mean();loss.backward()
    assert model.head[-1].bias.grad.abs().sum()>0


def test_campaign_replay_komi_and_saved_prediction():
    from blokus.console import Sources,campaign_games
    paths=campaign_games('train')
    if not paths:return
    data=Sources('/unused','/unused').campaign('train',0)
    assert data['komi']==4 and data['predictions']['komi']==4
    assert all(x is None for x in data['predictions']['values'][:8])
    e=data['predictions']['values'][-1];scores=data['frames'][-1]['scores'];d=scores[0]-scores[1]-4
    assert e['actual']==d*(1 if e['player']==0 else -1)
    assert e['exact'] and e['actual']==e['adjusted']
