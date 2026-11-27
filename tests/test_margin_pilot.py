import json
import pytest
from blokus.engine import State,Move,play
from blokus.margin_value import MarginValue,features,FORMAT
from blokus.komi_analysis import summary


def model(tmp_path):
    p=tmp_path/'model.json';p.write_text(json.dumps(dict(format=FORMAT,weights=[0.0]*171)))
    return p


def test_raw_points_terminal_bonus_and_komi_view(tmp_path):
    s=State();v=MarginValue(model(tmp_path),4)
    assert v.raw(s)==0 and v(s)==-4
    s.turn=1;assert v(s)==4
    s.remaining=[0,0];s.last=[1,0];s.passes=2
    assert v.raw(s)==5 and v(s)==9
    s.turn=0;assert v.raw(s)==-5 and v(s)==-9


def test_features_antisymmetric_and_phase_shape():
    s=State();play(s,Move(0,0,4,4),validate=True)
    a=features(s);s.turn=1-s.turn;b=features(s)
    assert len(a)==171 and a==[-v for v in b]


def test_komi_regrade_draws_and_cluster_uncertainty():
    rows=[dict(margin=v,cluster=str(i)) for i,v in enumerate((0,4,10))]
    r=summary(rows,[x['margin']-4 for x in rows])
    assert (r['wins'],r['draws'],r['losses'])==(1,1,1)
    assert r['point_rate']==.5 and r['win_rate']==pytest.approx(1/3)
    assert r['mean']==pytest.approx(2/3) and r['median']==0
    twice=summary(rows+rows,[x['margin']-4 for x in rows+rows])
    assert twice['clusters']==3 and twice['mean_ci95']==r['mean_ci95']
    assert r['winning_margin_mean']==6
