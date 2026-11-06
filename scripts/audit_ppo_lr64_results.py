import json
from pathlib import Path
from blokus.ppo_lr64 import R,SEEDS,write
from blokus.ppo_lr64_eval import e
from blokus.ppo_gpu_train import sha
from blokus.tournament import summarize,read_prefixes
completion=json.loads((R/'training-complete.json').read_text());assert completion['seconds']<=3600 and completion['common_iteration']==5120
assert sum(x['games'] for x in completion['results'])==147456
for seed in SEEDS:
 forks=[x for x in completion['results'] if x['seed']==seed and x['start']==4096];assert len(forks)==3 and len({x['base']['sha256'] for x in forks})==1
train=json.loads((R/'training-audit.json').read_text());assert sum(x['games'] for x in train['rows'] if x['main'])==147456 and train['shared_external_prefixes']==49152
for path,expected in json.loads((R/'source-manifest.json').read_text()).items():assert sha(path)==expected
lock=json.loads((R/'selection.json').read_text());end=lock['fixed_iteration'];assert end==json.loads((R/'training-complete.json').read_text())['common_iteration']
for row in lock['checkpoints'].values():assert sha(row['path'])==row['sha256']
for step,data in lock['curves'].items():
 for label,summary in data.items():
  matches=[(seed,json.loads((R/f'validation-{step}-{seed}-{label}.json').read_text())) for seed in SEEDS];assert e.cluster(matches)==summary
final=lock['curves'][str(end)];assert lock['gate']==(final['C-gen2']['point']>.5 and final['C-gen2']['difference']>=0 and final['C-B']['point_ci95'][0]>.45)
if (R/'holdout-summary.json').exists():
 assert lock['gate']
 for label,summary in json.loads((R/'holdout-summary.json').read_text()).items():assert e.cluster([(seed,json.loads((R/f'holdout-{seed}-{label}.json').read_text())) for seed in SEEDS])==summary
audit=json.loads((R/'evaluation-audit.json').read_text());assert audit['pass_'] and all(r['start_overlap']==0 and r['later_state_overlap']==0 for r in audit['rows'])
n=0
for phase in ('validation','holdout'):
 if not (R/f'{phase}-prefixes.json').exists():continue
 prefixes=read_prefixes(R/f'{phase}-prefixes.json');ids={p['id'] for p in prefixes['prefixes']}
 for path in R.glob(f'{phase}-*.json'):
  d=json.loads(path.read_text())
  if 'checkpoint' not in d:continue
  assert summarize(d['games'])==d['summary'];assert {(g['prefix_id'],g['seat']) for g in d['games']}=={(i,s) for i in ids for s in (0,1)};assert len(d['games'])==2*len(ids)
  assert d['left_ms']>0 and d['right_ms']>0 and d['wall']>0;n+=len(d['games'])
write(R/'arithmetic-audit.json',dict(pass_=True,games=n,selection_fixed=True,gate_recomputed=True,all_cluster_bootstraps_recomputed=True,checkpoint_sha_verified=True))
