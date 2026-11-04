"""Research models and bounded search; engine semantics are deliberately untouched."""
import copy
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .engine import State, Move, PLACEMENTS, START, cells_of, legal_moves, play, frontier
from .ai import DIM, STATE_DIM, Net, features, heuristic, load, seed_all

ACTION_IDS = {q.move: i for i, q in enumerate(PLACEMENTS)}
RESEARCH_STATE_DIM = STATE_DIM + 3
PASS_ID = len(PLACEMENTS)
CELL_INDEX = np.zeros((PASS_ID, 5), dtype=np.int64)
CELL_WEIGHT = np.zeros((PASS_ID, 5), dtype=np.float32)
for i, q in enumerate(PLACEMENTS):
    for j, (r, c) in enumerate(q.move.cells):
        CELL_INDEX[i, j] = r * 14 + c
        CELL_WEIGHT[i, j] = 1 / len(q.move.cells)


def clone(s):
    return State(s.boards.copy(), s.remaining.copy(), s.last.copy(), s.turn, s.passes)


def encode_board(s):
    b = np.zeros((4, 14, 14), dtype=np.float32)
    for channel, p in enumerate((s.turn, 1-s.turn)):
        for r, c in cells_of(s.boards[p]): b[channel, r, c] = 1
        b[channel+2, START[p][0], START[p][1]] = 1
    return b


def action_mask(s):
    out = np.zeros(PASS_ID+1, dtype=bool)
    if s.done: return out
    qs = legal_moves(s)
    if qs:
        out[[ACTION_IDS[q.move] for q in qs]] = True
    else: out[PASS_ID] = True
    return out


def terminal_value(s, perspective=None):
    p = s.turn if perspective is None else perspective
    return (s.score(p)-s.score(1-p))/109


def static_value(s):
    """Always side-to-move; exact normalized score at terminal."""
    if s.done: return terminal_value(s)
    f0, _ = frontier(s, s.turn); f1, _ = frontier(s, 1-s.turn)
    return terminal_value(s) + .6*(f0.bit_count()-f1.bit_count())/109


class PolicyValue(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.cnn = method in ('B', 'C')
        if self.cnn:
            self.conv = nn.Sequential(nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(),
                                      nn.Conv2d(16, 24, 3, padding=1), nn.ReLU())
            self.global_net = nn.Sequential(nn.Flatten(), nn.Linear(24*14*14, 64), nn.ReLU())
            extra = 64 + 24
        else: extra = 0
        self.policy = nn.Sequential(nn.Linear(DIM+RESEARCH_STATE_DIM+extra, 96), nn.Tanh(), nn.Linear(96, 1))
        self.value = nn.Sequential(nn.Linear(RESEARCH_STATE_DIM+(64 if self.cnn else 0), 64), nn.Tanh(), nn.Linear(64, 1), nn.Tanh())
        # Identical known heuristic prior for all architectures, explicitly reported.
        nn.init.zeros_(self.policy[-1].weight); nn.init.zeros_(self.policy[-1].bias)
        self.register_buffer('prior_weights', torch.tensor([12,4,2,1.2,2,.3,1,.2,-.6,0,0,.2], dtype=torch.float32))
        self.register_buffer('cell_index', torch.from_numpy(CELL_INDEX), persistent=False)
        self.register_buffer('cell_weight', torch.from_numpy(CELL_WEIGHT), persistent=False)

    def forward(self, board, state, x, ids):
        st = state.reshape(1, -1)
        parts = [x, st.expand(len(x), -1)]
        if self.cnn:
            spatial = self.conv(board.unsqueeze(0))
            global_features = self.global_net(spatial)
            local = (spatial[0].flatten(1)[:, self.cell_index[ids]] * self.cell_weight[ids].unsqueeze(0)).sum(-1).T
            parts += [global_features.expand(len(x), -1), local]
            st = torch.cat((st, global_features), 1)
        logits = x @ self.prior_weights + self.policy(torch.cat(parts, 1)).squeeze(-1)
        return logits, self.value(st).squeeze()


def pack(s, qs):
    x, st = features(s, qs)
    st = np.concatenate((st, np.array([s.score(s.turn)/89, s.score(1-s.turn)/89, s.passes/2], dtype=np.float32)))
    return encode_board(s), st, x, np.array([ACTION_IDS[q.move] for q in qs], dtype=np.int64)


def to_device(data, device):
    return tuple(torch.as_tensor(a, device=device) for a in data)


def search(s, logits=None, width=4, reply_width=3):
    """2-ply beam negamax, root prior from student, replies from heuristic.

    Returns (legal placements, searched indices, root-perspective values, nodes).
    No input mutation; every generated child, including pass, counts as a node.
    Width-limited search does NOT prove a minimax optimum over all legal moves.
    """
    if width < 1 or reply_width < 1: raise ValueError('positive beam widths required')
    qs = legal_moves(s)
    if s.done: return qs, [], np.empty(0), 0
    if not qs:
        child = clone(s); play(child, None, validate=False)
        return [], [], np.array([-static_value(child)]), 1
    if logits is None:
        x, _ = features(s, qs); logits = heuristic(x)
    selected = np.argsort(-np.asarray(logits), kind='stable')[:width].tolist()
    values = []; nodes = 0
    for i in selected:
        child = clone(s); play(child, qs[i].move, validate=False); nodes += 1
        if child.done: values.append(-static_value(child)); continue
        replies = legal_moves(child)
        if not replies:
            leaf = clone(child); play(leaf, None, validate=False); nodes += 1
            # Two turns have elapsed: leaf is in root player's perspective.
            values.append(static_value(leaf)); continue
        rx, _ = features(child, replies)
        ri = np.argsort(-heuristic(rx), kind='stable')[:reply_width]
        reply_values = []
        for j in ri:
            leaf = clone(child); play(leaf, replies[int(j)].move, validate=False); nodes += 1
            reply_values.append(-static_value(leaf))  # opponent maximizes this
        values.append(-max(reply_values))
    return qs, selected, np.array(values), nodes


def source_hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(path, model, optimizer, episode, metadata):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ck = dict(format=2, method=model.method, model=model.state_dict(), optimizer=optimizer.state_dict(),
              episode=episode, metadata=metadata, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(ck, str(p)+'.tmp'); os.replace(str(p)+'.tmp', p)


def restore(path, model, optimizer=None, rng=False):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    if ck['format'] != 2 or ck['method'] != model.method: raise ValueError('incompatible checkpoint')
    model.load_state_dict(ck['model'])
    if optimizer is not None: optimizer.load_state_dict(ck['optimizer'])
    if rng:
        random.setstate(ck['python_rng']); np.random.set_state(ck['numpy_rng']); torch.set_rng_state(ck['torch_rng'])
        if ck['cuda_rng'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck['cuda_rng'])
    return ck


class Agent:
    def __init__(self, spec, device='cpu', width=4, reply_width=3):
        self.spec = spec; self.device = device; self.width = width; self.reply_width = reply_width
        self.search_on = spec.endswith(':search')
        path = spec[:-7] if self.search_on else spec
        self.model = None
        self.kind = path
        if path not in ('random', 'heuristic', 'human'):
            ck = torch.load(path, map_location='cpu', weights_only=False)
            if ck.get('format') == 2:
                self.model = PolicyValue(ck['method']).to(device); self.kind = 'research'
            else: self.model = Net().to(device); self.kind = 'v1'
            self.model.load_state_dict(ck['model']); self.model.eval()
        self.stats = dict(calls=0, legal_seconds=0., feature_seconds=0., network_seconds=0., search_seconds=0., total_seconds=0., nodes=0, candidates=0)

    def act(self, s, rng):
        if s.done: raise ValueError('terminal state')
        start = time.perf_counter(); t = start; qs = legal_moves(s)
        self.stats['legal_seconds'] += time.perf_counter()-t
        self.stats['calls'] += 1; self.stats['candidates'] += len(qs)
        if not qs: move = None
        elif self.kind == 'random': move = rng.choice(qs).move
        elif self.kind == 'human':
            for i, q in enumerate(qs): print(i, q.move, q.move.cells)
            while True:
                try:
                    i = int(input('move index> '))
                    if 0 <= i < len(qs): break
                except ValueError: pass
                print('Invalid index')
            move = qs[i].move
        else:
            t = time.perf_counter()
            if self.kind == 'research': data = pack(s, qs)
            else: x, st = features(s, qs)
            self.stats['feature_seconds'] += time.perf_counter()-t
            t = time.perf_counter()
            with torch.no_grad():
                if self.kind == 'research': scores = self.model(*to_device(data, self.device))[0].cpu().numpy()
                elif self.kind == 'v1': scores = self.model(torch.as_tensor(x, device=self.device), torch.as_tensor(st, device=self.device))[0].cpu().numpy()
                else: scores = heuristic(x)
            self.stats['network_seconds'] += time.perf_counter()-t
            if self.search_on:
                t = time.perf_counter(); _, selected, values, nodes = search(s, scores, self.width, self.reply_width)
                self.stats['nodes'] += nodes; self.stats['search_seconds'] += time.perf_counter()-t
                best = np.flatnonzero(values >= values.max()-1e-7)
                choice = selected[int(rng.choice(best))]
            else:
                best = np.flatnonzero(scores >= scores.max()-1e-7); choice = int(rng.choice(best))
            move = qs[choice].move
        self.stats['total_seconds'] += time.perf_counter()-start
        return move
