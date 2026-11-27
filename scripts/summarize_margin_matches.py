import json
from pathlib import Path
from collections import Counter
import numpy as np
from blokus.komi_analysis import analyze,summary
R=Path('artifacts/margin-pilot');games=[json.loads(p.read_text()) for p in sorted((R/'matches').glob('*.json'))]
assert len(games)==len(json.loads((R/'tasks.json').read_text()))==30
rows=[]
for d in games:
 t=d['task'];score=d['scores'];remaining=d['remaining']
 rows.append(dict(stage=0,phase='standard-self' if t['kind']=='self' else 'heldout-paired',left='gen4 / Frozen' if t['kind']=='self' else '点差実験版',right='gen4 / Frozen',seat=t['seat'],scores=score,remaining=remaining,margin=score[0]-score[1],unplaced_advantage=remaining[1]-remaining[0],budget_ms=500,max_depth=6,prefix_plies=len(t['prefix']),cluster=t['cluster'],seed=t['seed']))
a=analyze(rows);paired=[r for r in rows if r['phase']=='heldout-paired'];standard=[r for r in rows if r['phase']=='standard-self'];performance=summary(paired,[r['margin']*(1 if r['seat']==0 else -1) for r in paired]);std=summary(standard)
# Diagnostic timing is per actual method; budget is same, work per move can differ.
timing={}
for who in ('margin','gen4'):
 stats=[s for d in games if d['task']['kind']=='paired' for s in d['stats'] if s['agent']==who and not s['forced_pass']]
 v=np.array([s['elapsed_ms'] for s in stats]);timing[who]=dict(moves=len(v),mean_ms=float(v.mean()),p95_ms=float(np.quantile(v,.95)),max_ms=float(v.max()),over_550=int(sum(v>550)),depths=dict(Counter(s['depth'] for s in stats)))
unique=len({d['trace_sha256'] for d in games if d['task']['kind']=='self'})
candidates=[dict(k=k,**summary(standard,[r['margin']-k for r in standard])) for k in (0,1,2,3,4,5,6,8,10,12)]
best=min(candidates,key=lambda v:(abs(v['point_rate']-.5),abs(v['mean']),v['k']))
recommendation=f"標準初期盤面は3固定モデルseed×先後交換の6局、独立した軌跡は{unique}種類だけです。標準コミの本番推奨は保留。今回の再採点ではk={best['k']}点が候補中で得点率50%に最も近い（先手得点率{100*best['point_rate']:.1f}%、{best['draws']}分、中央値{best['median']:.1f}点）。小標本・ほぼ決定的な同一AIなので探索候補に留め、異なる独立条件での追加検証前に固定しません。"
text=f"同じ500ms/幅4/最大6手、未使用12prefixの先後交換24局: 点差実験版 {performance['wins']}勝{performance['draws']}分{performance['losses']}敗、勝率{100*performance['win_rate']:.1f}%、得点率{100*performance['point_rate']:.1f}%、平均点差{performance['mean']:.2f}。強化を断定できる規模ではありません。"
a.update(performance=performance,standard=std,standard_candidates=candidates,standard_unique_traces=unique,timing=timing,recommendation=recommendation,comparison_text=text)
(R/'match-analysis.json').write_text(json.dumps(a,ensure_ascii=False))
print(json.dumps(dict(performance=performance,timing=timing,recommendation=recommendation,comparison_text=text),ensure_ascii=False,indent=2))
