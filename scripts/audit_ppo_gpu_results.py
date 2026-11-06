"""Selection/arithmetic/call-time audit without generating new matches."""
import json,math
from pathlib import Path
import numpy as np
from blokus.ppo_gpu_eval import R,SEEDS,cluster
from blokus.ppo_gpu_train import sha
from blokus.tournament import read_prefixes
lock=json.loads((R/'selection.json').read_text());timings=[]
for seed in SEEDS:
 rows=[r for r in lock['curves'] if r['seed']==seed]
 for row in rows:
  for label in ('untrained','heuristic','gen2'):
   d=json.loads((R/f"validation-{seed}-{row['step']}-{label}.json").read_text());assert row['scores'][label]==d['summary']['point_rate']
  assert row['criterion']==.5*row['scores']['untrained']+.25*row['scores']['heuristic']+.25*row['scores']['gen2']
 best=max(rows,key=lambda x:(x['criterion'],-x['step']));chosen=lock['selected'][str(seed)];assert chosen['iteration']==best['step'] and chosen['criterion']==best['criterion'] and chosen['sha256']==sha(chosen['path'])
for phase in ('validation','holdout'):
 prefix=read_prefixes(R/f'{phase}-prefixes.json');mapping={p['id']:len(p['moves'])%2 for p in prefix['prefixes']}
 for path in sorted(R.glob(f'{phase}-*.json')):
  d=json.loads(path.read_text())
  if 'games' not in d:continue
  left=right=0
  for g in d['games']:
   left+=sum((mapping[g['prefix_id']]+i)%2==g['seat'] for i in range(len(g['moves'])));right+=sum((mapping[g['prefix_id']]+i)%2!=g['seat'] for i in range(len(g['moves'])))
  measured=(left*d['left_ms']+right*d['right_ms'])/1000;assert np.isfinite(measured) and 0<measured<=d['wall']+.001
  timings.append(dict(path=str(path),left_calls=left,right_calls=right,act_seconds=measured,wall=d['wall']))
summary=json.loads((R/'holdout-summary.json').read_text())
for label in ('untrained','heuristic','gen2'):
 matches=[(s,json.loads((R/f'holdout-{s}-{label}.json').read_text())) for s in SEEDS];assert cluster(matches)==summary[label]
 allg=[g for _,r in matches for g in r['games']];assert math.isclose(summary[label]['point'],sum((g['difference']>0)+.5*(g['difference']==0) for g in allg)/len(allg));assert math.isclose(summary[label]['difference'],sum(g['difference'] for g in allg)/len(allg))
(R/'result-arithmetic-audit.json').write_text(json.dumps(dict(pass_=True,selection_maximum_and_hash=True,cluster_summary_recomputed=True,all_points_scores_recomputed=True,timings=timings),indent=2));print('RESULT AUDIT PASS',len(timings))
