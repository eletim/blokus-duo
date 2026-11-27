import gzip
import json
import sqlite3
import pytest
from blokus.console import Sources, frames


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, 'wt') if path.suffix == '.gz' else path.open('w')) as f:
        json.dump(data, f)


def test_missing_data_is_unknown_and_does_not_create_archive(tmp_path):
    source = Sources(tmp_path/'artifacts', tmp_path/'missing/games.sqlite3')
    data = source.catalog()
    assert data['groups'] == data['evaluations'] == data['training'] == data['archive'] == []
    assert data['progress']['generated'] is None
    assert not source.archive.exists()


def test_replay_includes_prefix_and_respects_seat(tmp_path):
    root = tmp_path/'value-placement-49152/n49152'
    # Legal first monomino for each seat; the first is an evaluation prefix.
    prefix = [[0, 0, 4, 4]]
    moves = [[0, 0, 9, 9]]
    score = frames(prefix+moves)[-1]['scores']
    write(root/'confirmation-bank.json', {'prefixes': [{'moves': prefix}]})
    path = root/'confirmation-AF-1.json.gz'
    write(path, dict(left='A', right='F', seed=1, budget_ms=500,
                    games=[dict(prefix=0, seat=1, moves=moves, scores=score)]))
    source = Sources(tmp_path, tmp_path/'none.sqlite3')
    result = source.game(str(path.relative_to(tmp_path)), 0)
    assert result['players'][0] == 'gen3 / Frozen'
    assert result['prefix_plies'] == 1
    assert len(result['frames']) == 3
    assert result['frames'][1]['boards'] == [[(4, 4)], []]
    assert result['frames'][2]['boards'] == [[(4, 4)], [(9, 9)]]
    with pytest.raises(KeyError): source.game('../../etc/passwd', 0)
    with pytest.raises(KeyError): source.game(str(path.relative_to(tmp_path)), -1)
    with pytest.raises(ValueError): frames(prefix+moves, [0, 0])
    with pytest.raises((ValueError, AssertionError)): frames(prefix+prefix)


def test_archive_filters_tests_and_never_writes(tmp_path):
    path = tmp_path/'games.sqlite3'
    moves = [[0, 0, 4, 4]]
    record = dict(generation='gen4', human_seat=1, moves=[dict(move=moves[0])],
                  scores=frames(moves)[-1]['scores'], status='unfinished')
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE games(id TEXT, started TEXT, mode TEXT, kind TEXT, record TEXT)')
        for key, kind in [('real', 'human'), ('test', 'test')]:
            db.execute('INSERT INTO games VALUES(?,?,?,?,?)', (key, '2026-11-20', 'gen4', kind, json.dumps(record)))
    before = path.read_bytes()
    source = Sources(tmp_path, path)
    assert [r['id'] for r in source.catalog()['archive']] == ['real']
    assert source.human_game('real')['players'] == ['gen4', '人間']
    with pytest.raises(KeyError): source.human_game('test')
    assert before == path.read_bytes()
