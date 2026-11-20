"""Frozen gen4 composite inference; no training history or optimizer required."""
import hashlib
import json
from pathlib import Path
import torch
from .ppo_policy import PPOAgent
from .ppo_search_deeper import DeepSearch
from .placement_value import PlacementValue, PlacementEvaluator
ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'models/gen4/manifest.json'

def manifest():
    return json.loads(MANIFEST.read_text())

def verify():
    m = manifest()
    for row in m['models']:
        for component in ('policy', 'value'):
            f = row[component]; data = (ROOT/f['path']).read_bytes()
            if len(data) != f['bytes'] or hashlib.sha256(data).hexdigest() != f['sha256']:
                raise ValueError(f'gen4 {component} hash mismatch: seed {row["seed"]}')
    return m

def load(seed=1751401, backend='auto'):
    m = verify(); rows = {x['seed']: x for x in m['models']}
    if seed not in rows:
        raise ValueError('gen4 seed must be one of 1751401, 1751402, 1751403')
    if backend == 'auto':
        from .ppo_native import library
        backend = 'native' if library() is not None else 'python'
    if backend not in ('python', 'native'):
        raise ValueError('backend must be auto, python or native')
    ck = torch.load(ROOT/rows[seed]['value']['path'], map_location='cpu', weights_only=True)
    if (ck.get('format'), ck.get('variant'), ck.get('seed'), ck.get('epochs'), ck.get('train_games')) != ('placement-value-v1', 'A', seed, 5, 49152):
        raise ValueError('gen4 checkpoint metadata mismatch')
    if ck['source_sha'] != rows[seed]['policy']['sha256']:
        raise ValueError('gen4 policy/value source mismatch')
    policy = PPOAgent(seed, 'cpu', generation='gen3', backend=backend)
    value = PlacementValue(policy.model, 'A'); value.load_state_dict(ck['model'])
    return policy, PlacementEvaluator(value, backend)

def search(pair):
    policy, value = pair
    return DeepSearch(policy, budget_ms=500, width=4, max_depth=6, value_evaluator=value)
