"""Audit grouped data, selection, paired scores/time and seed/prefix uncertainty."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_value import audit
from blokus.engine import State,Move,legal_moves,play
from blokus.ai import features,heuristic
from blokus.research import sha,clone
from blokus.value_search import encode
from blokus.ranking_data import initialize_worker,continuation
from blokus.value_experiment import write


def key(moves):
    s=State()
    for move in moves:play(s,Move(*move) if move else None)
    return (tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)


def main():
    root=Path(sys.argv[1] if len(sys.argv)>1 else 'artifacts/paired-ranking')
    out=root/'confirmed' if '--confirmed' in sys.argv else root
    fixture=json.loads((out/'holdout-prefixes.json').read_text());report=json.loads((out/'holdout-results.json').read_text())
    result=audit(out/'holdout-results.json',out/'holdout-prefixes.json')
    selection=json.loads((root/'selection.json').read_text())['selected']
    validation_results=json.loads((root/'validation-results.json').read_text())
    for seed,item in selection.items():
        for choice in item['choices']:
            values=[r['summary']['mean_difference'] for r in validation_results if r['seed']==int(seed) and r['alpha']==choice['spec']['alpha']]
            assert len(values)==2 and abs(np.mean(values)-choice['criterion'])<1e-12
        assert item['criterion']==max(x['criterion'] for x in item['choices'])
        assert sha(item['spec']['checkpoint'])==item['checkpoint_sha256']
    # Compare actual beginning states, including different move orders.
    train={key(x) for x in json.loads((root/'groups.prefixes.json').read_text())}
    for old in ['run/games.prefixes.json','run/branches.prefixes.json','compact/expanded.prefixes.json']:
        train.update(key(x) for x in json.loads((Path('artifacts/value-search')/old).read_text()))
    val={key(x['moves']) for x in json.loads((root/'validation-prefixes.json').read_text())['prefixes']}
    hold={key(x['moves']) for x in fixture['prefixes']}
    assert len(hold)==96 and not train&val and not train&hold and not val&hold
    old_fixtures=[Path('artifacts/value-search')/p for p in ['run/holdout1-prefixes.json','run/holdout2-prefixes.json','compact/holdout-prefixes.json','confirm/holdout-prefixes.json']]
    if out!=root:old_fixtures.append(root/'holdout-prefixes.json')
    for old in old_fixtures:
        assert not hold&{key(x['moves']) for x in json.loads(old.read_text())['prefixes']}
    d=np.load(root/'groups.npz');groups=json.loads((root/'groups.groups.json').read_text())
    assert len(groups)==len(d['game']) and np.isfinite(d['returns']).all()
    assert sha(root/'groups.npz')==json.loads((root/'groups.json').read_text())['sha256']
    # Replay a deterministic distributed subset, including actual terminal labels.
    initialize_worker()
    from blokus.ranking_data import AGENTS
    checked=0
    for i in np.linspace(0,len(groups)-1,min(32,len(groups)),dtype=int):
        g=groups[i];s=State(**g['root']);before=clone(s)
        for j,m in enumerate(g['moves']):
            child=clone(s);play(child,Move(*m));qs=legal_moves(child)
            if qs:x,_=features(child,qs);reply=qs[int(np.argmax(heuristic(x)))].move
            else:reply=None
            play(child,reply);b,x=encode(child)
            assert np.array_equal(np.packbits(b.reshape(-1)),d['board'][i,j]) and np.array_equal(x,d['state'][i,j])
            for r,seed in enumerate(g['continuation_seeds']):
                value,_=continuation(child,AGENTS[r%2],seed)
                assert abs(value-float(d['returns'][i,j,r]))<1e-7;checked+=1
        assert s==before
    estimates={};rng=np.random.default_rng(992211)
    for opponent in ['static','prior']:
        rows=[]
        for r in report['results']:
            if r['names'][0].startswith('ranking-') and r['names'][1]==opponent:
                rows.append([np.mean([g['difference'] for g in r['games'] if g['prefix_id']==p['id']]) for p in fixture['prefixes']])
        a=np.array(rows);assert a.shape==(3,96)
        si=rng.integers(3,size=(10000,3));pi=rng.integers(96,size=(10000,96));means=a[si[:,:,None],pi[:,None,:]].mean((1,2))
        estimates[opponent]=dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),seeds=3,prefixes=96,method='crossed seed/prefix bootstrap after averaging paired seats',provisional=True)
    result['timing_validity']='repaired scheduler; inspect measured times' if out!=root else 'diagnostic only: ranking-303/prior affected by estimator starvation'
    result.update(selection='PASS',data_label_replays=checked,train_eval_state_disjoint=True,crossed_bootstrap=estimates)
    write(out/'audit.json',result)
    lines=['# 同一局面候補の差分・ランキング学習','',
           '共通6ms予算の逐次2ply探索。実消費時間は完全一致ではない。各組96prefix×先後交換=192局。',
           '旧holdoutは選択に使わず、新validationで固定後fresh holdoutを生成。時間推定修正版は重み/混合率固定のままseed230000へ移行。', '',
           '| 相手 | 3seed平均得点差 | seed×prefix 95%区間 |','|---|---:|---:|']
    for name,v in estimates.items():lines.append(f"| {name} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |")
    lines+=['','| 左/右 | 勝/敗/分 | 得点差 [prefix CI] | 左/右ms/手 | 左/右超過率 |','|---|---:|---:|---:|---:|']
    for r in report['results']:
        s=r['summary'];a=r['left']['stats'];b=r['right']['stats'];lo,hi=s['difference_ci95']
        lines.append(f"| {' / '.join(r['names'])} | {s['wins']}/{s['losses']}/{s['draws']} | {s['mean_difference']:+.3f} [{lo:+.3f},{hi:+.3f}] | {a['milliseconds_per_move']:.3f}/{b['milliseconds_per_move']:.3f} | {a['overruns']/a['calls']:.2%}/{b['overruns']/b['calls']:.2%} |")
    lines+=['', '時間制御について: '+result['timing_validity'], '', 'prefix内2局を独立観測にせず、平均してから再標本化。seed×prefixでは共通prefix drawを3seed間で共有。3seedの学習不確実性推定は粗く、多重比較未補正。',
            'データ・checkpoint選択・得点符号・先後・時間・hashの工房側監査PASS。独立秘書レビューとは区別する。']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
