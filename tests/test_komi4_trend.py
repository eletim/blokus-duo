from scripts.komi4_trend import task,prefix,pair,SEEDS,baseline
from blokus.margin_search import key

def test_benchmark_conditions_same_across_generations_and_seed_openings():
    a=task('benchmark',3,5,'gen4',0,SEEDS[0]);b=task('benchmark',3,15,'gen4',0,SEEDS[0])
    assert {k:v for k,v in a.items() if k not in ['id','generation']}=={k:v for k,v in b.items() if k not in ['id','generation']}
    c=task('benchmark',3,5,'prior',1,SEEDS[2])
    sa,ma=prefix(a,pair(SEEDS[0])[0]);sc,mc=prefix(c,pair(SEEDS[2])[0])
    assert ma==mc and key(sa)==key(sc)
    assert a['prefix_seed']!=task('final',3)['prefix_seed']
    assert a['prefix_seed']!=task('train',3,6)['prefix_seed']
    assert a['budget_ms']==500 and a['width']==4 and a['max_depth']==6

def test_frozen_baseline_is_previous_g5():
    assert baseline(SEEDS[0])=='artifacts/komi4-scale/models/g5-1751401.pt'
