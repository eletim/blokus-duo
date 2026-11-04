"""Shared legal prefixes and cluster bootstrap evaluation."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from .engine import State, Move, legal_moves, play, is_legal
from .research import Agent, sha, source_hashes


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prefix_state(prefix):
    s = State()
    for row in prefix['moves']:
        move = Move(*row) if row is not None else None
        play(s, move)
    return s


def make_prefixes(seed, count):
    rng = random.Random(seed); prefixes = []; seen = set()
    while len(prefixes) < count:
        s = State(); moves = []
        for _ in range(rng.randint(2, 6)):
            qs = legal_moves(s); m = rng.choice(qs).move if qs else None
            moves.append([m.piece, m.orientation, m.row, m.col] if m else None)
            play(s, m)
        h = digest(moves)
        if h not in seen:
            seen.add(h); prefixes.append(dict(id=h, moves=moves))
    return dict(format=1, seed=seed, count=count, sha256=digest(prefixes), prefixes=prefixes)


def read_prefixes(path):
    data = json.loads(Path(path).read_text())
    if digest(data['prefixes']) != data['sha256']: raise ValueError('prefix set hash mismatch')
    if len(data['prefixes']) != data['count']: raise ValueError('prefix count mismatch')
    ids = set()
    for p in data['prefixes']:
        if digest(p['moves']) != p['id'] or p['id'] in ids: raise ValueError('prefix identity mismatch')
        ids.add(p['id']); prefix_state(p)
    return data


def summarize(games, seed=9817, draws=4000):
    grouped = {}
    for g in games: grouped.setdefault(g['prefix_id'], []).append(g)
    if any(len(v) != 2 or {g['seat'] for g in v} != {0,1} for v in grouped.values()):
        raise ValueError('each prefix must have both seats exactly once')
    diffs = np.array([np.mean([g['difference'] for g in v]) for v in grouped.values()])
    rates = np.array([np.mean([float(g['difference'] > 0)+.5*float(g['difference'] == 0) for g in v]) for v in grouped.values()])
    idx = np.random.default_rng(seed).integers(len(diffs), size=(draws, len(diffs)))
    ci_d = np.quantile(diffs[idx].mean(1), [.025,.975]).tolist()
    ci_w = np.quantile(rates[idx].mean(1), [.025,.975]).tolist()
    all_d = [g['difference'] for g in games]
    return dict(prefixes=len(diffs), games=len(games), wins=sum(d>0 for d in all_d), losses=sum(d<0 for d in all_d), draws=sum(d==0 for d in all_d),
                mean_difference=float(diffs.mean()), difference_ci95=ci_d, point_rate=float(rates.mean()), point_rate_ci95=ci_w,
                bootstrap=dict(unit='prefix pair, not individual game', samples=draws, seed=seed), provisional=True)


def match(left, right, prefix_set, device='cuda', seed=7781, width=4, reply_width=3):
    a = Agent(left, device, width, reply_width); b = Agent(right, device, width, reply_width)
    games = []; start = time.perf_counter()
    for i, prefix in enumerate(prefix_set['prefixes']):
        for seat in (0,1):
            s = prefix_state(prefix)
            # RNG attached to colour, shared across opponents and pair assignments.
            rngs = [random.Random(seed+i*2+p) for p in (0,1)]
            moves = 0
            while not s.done:
                agent = a if s.turn == seat else b
                move = agent.act(s, rngs[s.turn])
                if move is None:
                    if legal_moves(s): raise AssertionError('illegal pass')
                    play(s, None, validate=False)
                else: play(s, move)
                moves += 1
                if moves > 86: raise AssertionError('nonterminating game')
            games.append(dict(prefix_id=prefix['id'], seat=seat, scores=[s.score(0),s.score(1)],
                              difference=s.score(seat)-s.score(1-seat), plies_after_prefix=moves))
    def metadata(agent):
        path = agent.spec.removesuffix(':search')
        return dict(spec=agent.spec, checkpoint_sha256=sha(path) if Path(path).is_file() else None,
                    stats=agent.stats, milliseconds_per_move=1000*agent.stats['total_seconds']/max(1,agent.stats['calls']))
    return dict(left=metadata(a), right=metadata(b), prefix_sha256=prefix_set['sha256'], prefix_seed=prefix_set['seed'],
                summary=summarize(games), games=games, wall_seconds=time.perf_counter()-start,
                search=dict(width=width, reply_width=reply_width, depth=2, equal_thinking_time=False))


def tournament(specs, pairs, prefix_set, output, device='cuda'):
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    report = dict(prefix_sha256=prefix_set['sha256'], specs=specs, source_hashes=source_hashes(), results=[])
    for left, right in pairs:
        result = match(specs[left], specs[right], prefix_set, device)
        result['names'] = [left, right]; report['results'].append(result)
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps(dict(pair=[left,right], **result['summary'])), flush=True)
    return report
