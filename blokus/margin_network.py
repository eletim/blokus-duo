"""Independent nonlinear final-point regressor. Research rule: second +4.

The output is a terminal score margin in mover points, not a win probability.
Only the representation is initialized from a frozen policy; targets are scores.
"""
import copy
import numpy as np
import torch
from torch import nn
from .value_inputs import encode
from .placement_value import mobility

KOMI = 4.

def terminal_margin(s):
    d = s.score(0) - s.score(1) - KOMI
    return float(d if s.turn == 0 else -d)

def inputs(s):
    b, st = encode(s)
    return b[:4], np.append(st, 1. if s.turn == 0 else -1.).astype(np.float32), mobility(s)

class MarginNet(nn.Module):
    def __init__(self, source=None):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(4,8,3,padding=1),nn.ReLU(),nn.Conv2d(8,8,3,padding=1),nn.ReLU())
        self.global_net = nn.Sequential(nn.Flatten(),nn.Linear(8*196,32),nn.Tanh())
        if source is not None:
            self.conv = copy.deepcopy(source.conv)
            self.global_net = copy.deepcopy(source.global_net)
        self.mobility_head = nn.Sequential(nn.Linear(252,64),nn.ReLU())
        self.head = nn.Sequential(nn.Linear(146,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,b,st,extra):
        g = self.global_net(self.conv(b.float()))
        # Residual target is learned from terminal scores; current score is only a feature.
        current = (st[:,46]-st[:,47])*89 - KOMI*st[:,49]
        return self.head(torch.cat((g,st,self.mobility_head(extra)),1)).squeeze(-1)*32 + current

class NetworkEvaluator:
    def __init__(self,path):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        assert saved['komi'] == KOMI
        self.model = MarginNet(); self.model.load_state_dict(saved['model']); self.model.eval()
    def __call__(self,s):
        if s.done:return terminal_margin(s)
        b,st,e=inputs(s)
        with torch.inference_mode():
            return float(self.model(torch.as_tensor(b[None]),torch.as_tensor(st[None]),torch.as_tensor(e[None])).item())
