"""Non-neural candidate features used by fixed linear inference."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
from .engine import PLACEMENTS, State, legal_moves, play

ACTIVE = [0,1,2,3,4,5,6,7,8,11]  # drop action-independent remaining/difference
PRIOR = np.array([12,4,2,1.2,2,.3,1,.2,-.6,.2], dtype=np.float64)
FORMAT = 'phase-linear-reinforce-v1'


def inputs(s, backend='python'):
    if backend == 'native':
        from .ppo_native import scan
        ids, x = scan(s)
        qs = [PLACEMENTS[i] for i in ids]
    elif backend == 'python':
        from .ai import features
        qs = legal_moves(s)
        x, _ = features(s, qs)
    else:
        raise ValueError('backend must be python or native')
    remaining = s.remaining[s.turn].bit_count()
    phase = 0 if remaining > 14 else 1 if remaining > 7 else 2
    return np.asarray(x[:, ACTIVE], dtype=np.float64), qs, phase


def probabilities(x, weights):
    logits = x @ weights
    p = np.exp(logits-logits.max())
    return p/p.sum()
