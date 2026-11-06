"""Shared fixed linear-L model and completed-stage search for CLI and Web."""
from pathlib import Path
from .linear_value import Agent
from .linear_tree import CompletedSearch

MODEL_SEEDS = (3101, 3102, 3103)

def load_search(model_seed=3101, backend='auto', checkpoint=None):
    if model_seed not in MODEL_SEEDS:
        raise ValueError('非NN Lのseedは3101/3102/3103です')
    if backend == 'auto':
        from .ppo_native import library
        backend = 'native' if library() is not None else 'python'
    if backend not in ('python', 'native'):
        raise ValueError('backend must be python, native, or auto')
    path = Path(checkpoint) if checkpoint else Path(__file__).resolve().parents[1]/'models'/'linear-L'/f'seed-{model_seed}.json'
    return make_search(Agent.load(path, backend))

def make_search(agent):
    return CompletedSearch(agent, ms=100, width=24, replies=8, depth=2)
