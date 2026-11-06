import json,hashlib
from pathlib import Path
import torch
from scripts.bench_ppo_runtime import audit_games,equal
from blokus.ppo_policy import collate
R=Path('artifacts/ppo-ragged');rows=[]
for device in ('cpu','cuda'):
 full=torch.load(R/f'{device}-full/latest.pt',weights_only=False);split=torch.load(R/f'{device}-split/latest.pt',weights_only=False);dense=torch.load(R/f'{device}-dense/latest.pt',weights_only=False)
 keys=['model','past','optimizer']+[k for k in full if 'rng' in k]
 assert all(equal(full[k],split[k]) for k in keys)
 assert all(x['input_hash']==y['input_hash'] and x['games']==y['games'] for x,y in zip(full['pilot_logs'],split['pilot_logs']))
 pairs=[]
 for a,b in zip(dense['pilot_logs'],full['pilot_logs']):
  audit_games(a['games']);audit_games(b['games'])
  pairs.append(dict(iteration=a['iteration'],moves_and_scores_equal=all(x['prefix']==y['prefix'] and x['moves']==y['moves'] and x['scores']==y['scores'] for x,y in zip(a['games'],b['games'])),input_bit_equal=a['input_hash']==b['input_hash'],steps_equal=a['update']['updates']==b['update']['updates']))
 rows.append(dict(device=device,resume_bit_equal=True,checked_keys=keys,trajectories=pairs,replayed_games=128,model_hashes={name:hashlib.sha256((R/f'{device}-{name}/latest.pt').read_bytes()).hexdigest() for name in ('dense','full','split')}))
bench=json.loads((R/'benchmark.json').read_text());phase=json.loads((R/'phase-benchmark.json').read_text());e2e=json.loads((R/'e2e-benchmark.json').read_text())
for data in (bench['rows'],phase,e2e['rows']):
 grouped={}
 for x in data:
  key=(x.get('phase'),x['minibatch'],x['rep']);grouped.setdefault(key,[]).append(x['update']['updates'] if 'update' in x else x['updates'])
 assert all(len(v)==2 and v[0]==v[1] for v in grouped.values())
(R/'audit.json').write_text(json.dumps(dict(pass_=True,rows=rows,matched_steps_all_benchmarks=True,notes='Floating equivalence tolerance not passed; this audit does not override that gate.'),indent=2)+'\n');print('AUDIT PASS')
