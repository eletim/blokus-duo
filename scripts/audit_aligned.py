"""Selection, shared budgets, fresh-prefix and paired uncertainty audit."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_value import audit
from scripts.audit_ranking import key
from blokus.research import sha
from blokus.value_experiment import write


def main():
    root=Path('artifacts/search-aligned');report=json.loads((root/'holdout-results.json').read_text());fixture=json.loads((root/'holdout-prefixes.json').read_text())
    result=audit(root/'holdout-results.json',root/'holdout-prefixes.json');selection=json.loads((root/'selection.json').read_text());validation=json.loads((root/'validation-results.json').read_text())
    choices=selection['choices'];chosen=selection['chosen'];assert chosen['criterion']==max(c['criterion'] for c in choices)
    diagnostics={};hashes=set();steps=set();initials=set()
    for c in choices:
        values=[r['summary']['mean_difference'] for r in validation if r['mode']==c['mode']]
        assert len(values)==6 and abs(np.mean(values)-c['criterion'])<1e-12
        for seed,spec in c['specs'].items():
            folder=Path(spec['checkpoint']).parent;ck=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False)
            log=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()];best=min(log,key=lambda x:(x['decision_regret_points'],x['pair_mse']))
            assert ck['step']==best['step'];config=ck['config'];hashes.add(config['dataset_sha256']);initials.add(config['initialize_sha256'])
            summary=json.loads((folder/'summary.json').read_text());steps.add(summary['invocation_steps']);assert config['seed']==int(seed)
            diagnostics[f"{c['mode']}-{seed}"]=best
    assert len(hashes)==len(initials)==len(steps)==1 and steps=={6000}
    for seed,spec in chosen['specs'].items():assert sha(spec['checkpoint'])==selection['checkpoint_hashes'][seed]
    train=set()
    for p in [root/'groups.prefixes.json',root/'pilot.prefixes.json',Path('artifacts/paired-ranking/groups.prefixes.json')]+[Path('artifacts/value-search')/p for p in ['run/games.prefixes.json','run/branches.prefixes.json','compact/expanded.prefixes.json']]:train.update(key(x) for x in json.loads(p.read_text()))
    val={key(x['moves']) for x in json.loads((root/'validation-prefixes.json').read_text())['prefixes']};hold={key(x['moves']) for x in fixture['prefixes']}
    assert len(hold)==96 and not train&hold and not train&val and not val&hold
    for p in [Path('artifacts/paired-ranking/holdout-prefixes.json'),Path('artifacts/paired-ranking/confirmed/holdout-prefixes.json')]+[Path('artifacts/value-search')/p for p in ['run/holdout1-prefixes.json','run/holdout2-prefixes.json','compact/holdout-prefixes.json','confirm/holdout-prefixes.json']]:
        assert not hold&{key(x['moves']) for x in json.loads(p.read_text())['prefixes']}
    estimates={};rng=np.random.default_rng(332211)
    for opponent in ['static','prior']:
        rows=[]
        for r in report['results']:
            if r['names'][0].startswith('aligned-') and r['names'][1]==opponent:
                rows.append([np.mean([g['difference'] for g in r['games'] if g['prefix_id']==p['id']]) for p in fixture['prefixes']])
        a=np.array(rows);assert a.shape==(3,96);si=rng.integers(3,size=(10000,3));pi=rng.integers(96,size=(10000,96))
        means=a[si[:,:,None],pi[:,None,:]].mean((1,2));estimates[opponent]=dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),seeds=3,prefixes=96,method='paired seats averaged; crossed training-seed/prefix bootstrap',provisional=True)
    result.update(crossed_bootstrap=estimates,selection='PASS',shared_data_initial_checkpoint_updates='PASS',train_eval_state_disjoint=True,selected_mode=chosen['mode'])
    write(root/'audit.json',result);write(root/'selected-predictive-metrics.json',diagnostics)
    lines=['# 実探索に合わせたgroupと損失比較','',f"validation選択方式: **{chosen['mode']}**。32prefix、各方式3seed×2相手の平均得点差で選択。",'',
           '| 方式 | validation平均得点差 |','|---|---:|']
    for c in choices:lines.append(f"| {c['mode']} | {c['criterion']:+.3f} |")
    lines+=['','fresh holdout320000、96prefix×先後交換。共通6ms予算、実時間の完全一致ではない。', '',
            '| 相手 | 選択方式3seed平均得点差 | seed×prefix 95%区間 |','|---|---:|---:|']
    for name,v in estimates.items():lines.append(f"| {name} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] |")
    lines+=['','| 左/右 | 勝/敗/分 | 得点差 [prefix CI] | 左/右ms/手 | 左/右超過率 |','|---|---:|---:|---:|---:|']
    for r in report['results']:
        s=r['summary'];a=r['left']['stats'];b=r['right']['stats'];lo,hi=s['difference_ci95']
        lines.append(f"| {' / '.join(r['names'])} | {s['wins']}/{s['losses']}/{s['draws']} | {s['mean_difference']:+.3f} [{lo:+.3f},{hi:+.3f}] | {a['milliseconds_per_move']:.3f}/{b['milliseconds_per_move']:.3f} | {a['overruns']/a['calls']:.2%}/{b['overruns']/b['calls']:.2%} |")
    lines+=['','prefix内2局を独立標本としない。共通prefix drawを3seedで共有して再標本化。3seedは同一データ・事前学習を共有する更新乱数3seedであり、全学習工程の独立反復ではない。多重比較未補正、相手pool/96prefix限定。工房側監査であり秘書の独立監査とは区別する。']
    (root/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
