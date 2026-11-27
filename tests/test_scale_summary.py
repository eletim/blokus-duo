from scripts.summarize_komi4_scale import row,summarize

def test_score_bonus_komi_and_second_seat_perspective():
    g=dict(task=dict(id='x',split='standard',group='seed1',seed=1,seat=1,opponent='ridge'),scores=[20,-3],remaining=[0,3],adjusted_first_margin=19,trace_sha256='a')
    r=row(g)
    assert (r['margin'],r['unplaced_advantage'],r['bonus_delta'])==(-19,-3,-20)
    assert r['group']=='standard-initial'
    assert summarize([r])['ci95']['margin'] is None

def test_draws_and_paired_groups_are_not_extra_independent_openings():
    rows=[dict(group='a',trace=str(i),margin=d,unplaced_advantage=d,bonus_delta=0) for i,d in enumerate([4,-2,0])]
    r=summarize(rows)
    assert r['groups']==1 and r['games']==3
    assert r['wins']==r['draws']==r['losses']==1
    assert r['win_rate']==1/3 and r['point_rate']==.5
    assert r['mean_winning_margin']==4 and r['median_margin']==0
    assert r['ci95']['points'] is None
