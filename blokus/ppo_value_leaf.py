"""Candidate-free value inference. Same board/state and value arithmetic as PPO forward."""
import numpy as np
import torch
from .value_search import encode

def value_inputs(state,channels):
    if channels not in (4,6):raise ValueError('channels must be 4 or 6')
    board,features=encode(state)
    return board[:channels].copy(),features.copy()

def value_only(policy,state):
    if state.done:
        d=state.score(state.turn)-state.score(1-state.turn)
        return float((d>0)-(d<0))
    board,features=value_inputs(state,policy.channels)
    with torch.inference_mode():
        b=torch.as_tensor(board[None],device=policy.device)
        st=torch.as_tensor(features[None],device=policy.device)
        f=policy.model.conv(b.float());g=policy.model.global_net(f)
        return policy.model.value(torch.cat((g,st),-1)).squeeze(-1).item()
