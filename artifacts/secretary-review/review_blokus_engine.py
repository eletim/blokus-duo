"""Run against a copied engine or remote engine with PYTHONPATH configured."""
import random
import sys
import time
import importlib.util
import blokus_reference as ref

spec = importlib.util.spec_from_file_location('reviewed_engine', sys.argv[1])
e = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = e
spec.loader.exec_module(e)
assert {min(ref.orientations(p)) for p in e.PIECES} == ref.pieces()
rng = random.Random(29051)
snapshots = turns = 0
t = time.time()
for game in range(40):
    s = e.State()
    for turn in range(90):
        moves = e.legal_moves(s)
        if (game < 8 and turn % 7 == 0) or (game < 4 and not moves):
            own = set(e.cells_of(s.boards[s.turn]))
            opp = set(e.cells_of(s.boards[1-s.turn]))
            expected = {(i, c) for i, p in enumerate(e.PIECES)
                        if s.remaining[s.turn] >> i & 1
                        for c in ref.legal_cells(p, own, opp, ref.STARTS[s.turn])}
            actual = {(q.move.piece, tuple(sorted(q.move.cells))) for q in moves}
            assert expected == actual, (game, turn, len(expected), len(actual))
            snapshots += 1
        if s.done:
            break
        m = rng.choice(moves).move if moves else None
        assert e.is_legal(s, m)
        e.play(s, m)
        turns += 1
        assert not s.boards[0] & s.boards[1]
        for p in range(2):
            remaining_area = sum(len(e.PIECES[i]) for i in range(21)
                                 if s.remaining[p] >> i & 1)
            assert s.boards[p].bit_count() + remaining_area == 89
            assert s.score(p) == (-remaining_area if remaining_area else
                                  20 if s.last[p] == 0 else 15)
    assert s.done, 'Game did not terminate within the maximum length'
print({'games': 40, 'turns': turns, 'exhaustive_snapshots': snapshots,
       'seconds': round(time.time()-t, 2), 'result': 'PASS'})
