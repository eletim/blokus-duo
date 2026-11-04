"""Independent arithmetic/integrity review of paired tournament artifacts.

Run from the repository root: python scripts/audit_research.py REPORT PREFIXES
Uses the engine's independent coordinate legality checker, not tournament helpers.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blokus.engine import State, Move, play


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def audit(report_path, prefixes_path):
    report = json.loads(Path(report_path).read_text())
    fixture = json.loads(Path(prefixes_path).read_text())
    assert digest(fixture['prefixes']) == fixture['sha256'], 'fixture hash'
    assert report['prefix_sha256'] == fixture['sha256'], 'report fixture mismatch'
    ids = set()
    states = set()
    for prefix in fixture['prefixes']:
        assert digest(prefix['moves']) == prefix['id'], 'individual prefix hash'
        assert prefix['id'] not in ids, 'duplicate prefix'
        ids.add(prefix['id'])
        s = State()
        for row in prefix['moves']:
            play(s, Move(*row) if row is not None else None)
        state = (tuple(s.boards), tuple(s.remaining), tuple(s.last), s.turn, s.passes)
        assert state not in states, 'duplicate resulting state despite different histories'
        assert not s.done, 'terminal opening'
        states.add(state)
    assert len(ids) == fixture['count'], 'fixture count'
    audited = []
    for result in report['results']:
        assert result['prefix_sha256'] == fixture['sha256'], 'match fixture mismatch'
        groups = defaultdict(list)
        for game in result['games']:
            seat = game['seat']
            assert seat in (0, 1), 'invalid seat'
            assert game['difference'] == game['scores'][seat] - game['scores'][1-seat], 'score sign'
            assert 0 <= game['plies_after_prefix'] <= 86, 'game length'
            groups[game['prefix_id']].append(game)
        assert set(groups) == ids, 'missing or extra opening'
        for group in groups.values():
            assert len(group) == 2 and {g['seat'] for g in group} == {0, 1}, 'unpaired sample'
        diffs = [g['difference'] for g in result['games']]
        count = Counter('wins' if d > 0 else 'losses' if d < 0 else 'draws' for d in diffs)
        summary = result['summary']
        for key in ('wins', 'losses', 'draws'):
            assert summary[key] == count[key], f'incorrect {key}'
        assert summary['games'] == 2 * len(ids) and summary['prefixes'] == len(ids)
        assert abs(summary['mean_difference'] - statistics.mean(diffs)) < 1e-10
        rate = (count['wins'] + .5 * count['draws']) / len(diffs)
        assert abs(summary['point_rate'] - rate) < 1e-10
        pairs = [statistics.mean(g['difference'] for g in group) for group in groups.values()]
        lo, hi = summary['difference_ci95']
        assert min(pairs)-1e-10 <= lo <= hi <= max(pairs)+1e-10, 'CI outside bootstrap support'
        lo, hi = summary['point_rate_ci95']
        assert 0 <= lo <= hi <= 1, 'rate CI bounds'
        assert summary['bootstrap']['unit'] == 'prefix pair, not individual game'
        for side in ('left', 'right'):
            meta = result[side]
            path = Path(meta['spec'].removesuffix(':search'))
            if meta['checkpoint_sha256'] is not None:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['checkpoint_sha256'], 'checkpoint changed'
            st = meta['stats']
            assert st['calls'] > 0 and st['total_seconds'] >= 0
            expected_ms = 1000 * st['total_seconds'] / st['calls']
            assert abs(meta['milliseconds_per_move'] - expected_ms) < 1e-9
        audited.append({'names': result.get('names'), 'games': len(diffs), 'point_rate': rate})
    return {'result': 'PASS', 'prefixes': len(ids), 'unique_states': len(states), 'matches': audited}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report')
    parser.add_argument('prefixes')
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.prefixes), indent=2))
