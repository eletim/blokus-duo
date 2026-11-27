import pytest
from scripts.trend_paired_intervals import paired

def rows():
    return [dict(group=str(g),seed=1,seat=s,opponent='gen4',margin=(-1 if s==0 else 1)) for g in range(4) for s in [0,1]]

def test_pairs_cancel_identical_results_and_keep_opening_cluster():
    b=rows()
    x=paired(b,list(reversed(b)))
    assert x['groups']==4 and x['games']==8
    assert x['change']==dict(win=0,points=0,margin=0)
    assert x['ci95']['win']==[0,0]
    c=[dict(r,margin=r['margin']+2) for r in b]
    x=paired(b,c)
    assert x['change']==dict(win=.5,points=.5,margin=2)
    assert x['ci95']['margin']==[2,2]

def test_missing_or_duplicate_pair_is_rejected():
    b=rows()
    with pytest.raises(ValueError): paired(b,b[:-1])
    with pytest.raises(ValueError): paired(b,b+[b[0]])
