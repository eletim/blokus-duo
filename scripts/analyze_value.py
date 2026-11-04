"""Independent arithmetic, time accounting and crossed seed/prefix bootstrap."""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from blokus.engine import State,Move,play


def audit(path,fixture_path):
    report=json.loads(Path(path).read_text());fixture=json.loads(Path(fixture_path).read_text())
    def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert digest(fixture['prefixes'])==fixture['sha256']==report['prefix_sha256']
    ids=[];states=set()
    for prefix in fixture['prefixes']:
        assert digest(prefix['moves'])==prefix['id'];ids.append(prefix['id']);s=State()
        for row in prefix['moves']:play(s,Move(*row) if row is not None else None)
        key=(tuple(s.boards),tuple(s.remaining),tuple(s.last),s.turn,s.passes)
        assert key not in states;states.add(key)
    for result in report['results']:
        assert result['prefix_sha256']==fixture['sha256'];pairs=defaultdict(list);times=[[],[]]
        for g in result['games']:
            assert g['difference']==g['scores'][g['seat']]-g['scores'][1-g['seat']]
            pairs[g['prefix_id']].append(g)
            for who,ms in g['decision_ms']:
                assert who in [0,1] and ms>=0;times[who].append(ms)
        assert set(pairs)==set(ids)
        for gs in pairs.values():assert len(gs)==2 and {g['seat'] for g in gs}=={0,1}
        ds=[g['difference'] for g in result['games']];summary=result['summary']
        assert summary['wins']==sum(d>0 for d in ds) and summary['losses']==sum(d<0 for d in ds) and summary['draws']==sum(d==0 for d in ds)
        assert abs(summary['mean_difference']-np.mean(ds))<1e-10
        for who,side in enumerate(['left','right']):
            m=result[side];t=np.array(times[who]);st=m['stats']
            assert len(t)==st['calls'] and abs(t.mean()-st['milliseconds_per_move'])<1e-9
            assert abs(t.sum()/1000-st['total_seconds'])<1e-9
            assert int((t>result['budget_ms']).sum())==st['overruns']
            assert st['allowance_ms']==result['budget_ms']
            if m['spec'].get('checkpoint'):assert hashlib.sha256(Path(m['spec']['checkpoint']).read_bytes()).hexdigest()==m['checkpoint_sha256']
    estimates={};rng=np.random.default_rng(2027)
    for opponent in ['static','C']:
        rows=[];seeds=[]
        for r in report['results']:
            if r['names'][0].startswith('value-') and r['names'][1]==opponent:
                groups=defaultdict(list)
                for g in r['games']:groups[g['prefix_id']].append(g['difference'])
                rows.append([np.mean(groups[i]) for i in ids]);seeds.append(r['names'][0])
        if not rows:continue
        a=np.array(rows);si=rng.integers(len(rows),size=(4000,len(rows)));pi=rng.integers(len(ids),size=(4000,len(ids)))
        means=a[si[:,:,None],pi[:,None,:]].mean((1,2))
        estimates[opponent]=dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),seeds=seeds,prefixes=len(ids),
                                 method='crossed seed and prefix resampling; paired colours averaged first; shared prefix draw across seeds',provisional=True)
    return dict(audit='PASS',matches=len(report['results']),unique_prefixes=len(states),crossed_bootstrap=estimates)


def main():
    root=Path(sys.argv[1]);summaries={}
    candidates=[('stage1',root/'run/stage1/holdout-results.json',root/'run/holdout1-prefixes.json'),
                ('stage2',root/'run/stage2/holdout-results.json',root/'run/holdout2-prefixes.json'),
                ('compact',root/'compact/holdout-results.json',root/'compact/holdout-prefixes.json'),
                ('confirm',root/'confirm/holdout-results.json',root/'confirm/holdout-prefixes.json')]
    lines=['# 終局価値学習・同一時間予算探索の中間結果','',
           '全方式に生成/順位/GPU同期込み6msの同一allowanceを与えた。実時間をpaddingして揃えたものではない。',
           'native処理の非同期終了やOS待機による超過は記録し、deadline後に終了した探索結果は採用しない。', '',
           '| 段階 | 相手 | 3seed平均得点差 | seed×prefix bootstrap95%区間 |', '|---|---|---:|---:|']
    details=[]
    for name,path,fixture in candidates:
        if not path.exists():continue
        result=audit(path,fixture);summaries[name]=result
        for opponent,v in result['crossed_bootstrap'].items():lines.append(f"| {name} | {opponent} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] |")
        report=json.loads(path.read_text());details+=['',f'## {name} 個別結果と実時間','',
               '| 左/右 | 勝/敗/分 | 得点差 [prefix CI] | 左/右ms/手 | 左/右超過率 |', '|---|---:|---:|---:|---:|']
        for r in report['results']:
            s=r['summary'];a=r['left']['stats'];b=r['right']['stats'];lo,hi=s['difference_ci95']
            details.append(f"| {' / '.join(r['names'])} | {s['wins']}/{s['losses']}/{s['draws']} | {s['mean_difference']:+.2f} [{lo:+.2f},{hi:+.2f}] | {a['milliseconds_per_move']:.3f}/{b['milliseconds_per_move']:.3f} | {a['overruns']/a['calls']:.2%}/{b['overruns']/b['calls']:.2%} |")
    lines+=['', '3seed/24prefixの小規模診断。prefix内2局は平均してから集計し、共通prefixの相関を保ったままseedとprefixを再標本化した。3seedでは学習不確実性の推定自体が粗く、多数比較も未補正。',
            '段階ごとに別holdoutを使うため、段階間の点推定の差をそのまま改善量とは扱わない。checkpoint/混合率はvalidationで固定しholdoutから再選択していない。']
    lines+=details
    (root/'analysis.json').write_text(json.dumps(summaries,indent=2));(root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
