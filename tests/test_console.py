import gzip
import json
import sqlite3
import pytest
from blokus.console import Sources, frames


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with (gzip.open(path, 'wt') if path.suffix == '.gz' else path.open('w')) as f:
        json.dump(data, f)


def test_missing_data_is_unknown_and_does_not_create_archive(tmp_path, monkeypatch):
    monkeypatch.setattr("blokus.console.RESEARCH", tmp_path/"missing-research")
    monkeypatch.setattr("blokus.console.CAMPAIGN", tmp_path/"missing-campaign")
    monkeypatch.setattr("blokus.console.SCALE", tmp_path/"missing-scale")
    monkeypatch.setattr("blokus.trend_console.R", tmp_path/"missing-trend")
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
                    games=[dict(prefix=0, seat=1, moves=moves, scores=score, stats=[dict(player=1, root_scores=[-.2,.6], depth=2, stages=[2], fallback=False)])]))
    source = Sources(tmp_path, tmp_path/'none.sqlite3')
    result = source.game(str(path.relative_to(tmp_path)), 0)
    assert result['players'][0] == 'gen3 / Frozen'
    assert result['prefix_plies'] == 1
    assert result['evaluations'][0]['value'] is None
    assert result['evaluations'][1]['value'] == .6
    assert result['evaluations'][1]['player'] == 1
    assert result['evaluations'][2]['value'] is None
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


def test_recorded_selected_value_uses_completed_argmax_not_first_or_average():
    from blokus.console import recorded_evaluation
    stats = dict(player=1, root_scores=[-.8, .7, .2, .7], depth=4, stages=[2, 4], fallback=False)
    result = recorded_evaluation(stats, 1, 'gen4', [0, 0, 9, 9], trusted=True)
    assert result['value'] == .7 and result['root_index'] == 1
    assert result['player'] == 1 and result['depth'] == 4
    assert recorded_evaluation(stats, 0, 'gen4', [0,0,4,4], trusted=True)['value'] is None
    for extra in [dict(fallback=True), dict(forced_pass=True), dict(root_scores=[]),
                  dict(root_scores=[float('nan')]), dict(stages=[2])]:
        assert recorded_evaluation(stats | extra, 1, 'gen4', [0,0,9,9], trusted=True)['value'] is None
    assert recorded_evaluation(stats, 1, 'gen4', [0,0,9,9])['value'] is None


def test_score_rules_no_komi_and_full_placement_bonus():
    from blokus.engine import State
    s = State()
    assert [s.score(p) for p in (0,1)] == [-89, -89]
    s.remaining = [0,0];s.last = [1,0]
    assert [s.score(p) for p in (0,1)] == [15,20]
