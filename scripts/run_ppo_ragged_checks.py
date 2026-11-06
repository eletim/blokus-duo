import subprocess,json
from pathlib import Path
import torch
from scripts.bench_ppo_runtime import equal
R=Path('artifacts/ppo-ragged')
for device in ('cpu','cuda'):
 for mode,name,until,resume in [('dense','dense',4100,None),('compact','full',4100,None),('compact','split',4098,None),('compact','split',4100,str(R/f'{device}-split/latest.pt'))]:
  cmd=['.venv/bin/python','-m','scripts.pilot_ppo_ragged','--device',device,'--mode',mode,'--out',str(R/f'{device}-{name}'),'--until',str(until)]
  if resume:cmd+=['--resume',resume]
  subprocess.run(cmd,check=True)
 a=torch.load(R/f'{device}-full/latest.pt',weights_only=False);b=torch.load(R/f'{device}-split/latest.pt',weights_only=False);keys=['model','past','optimizer']+[k for k in a if 'rng' in k.lower()]
 assert all(equal(a[k],b[k]) for k in keys)
 ref=torch.load(R/f'{device}-dense/latest.pt',weights_only=False)
 rows=[]
 for x,y in zip(ref['pilot_logs'],a['pilot_logs']):
  rows.append(dict(iteration=x['iteration'],same_input=x['input_hash']==y['input_hash'],same_games=x['games']==y['games'],dense_updates=x['update']['updates'],compact_updates=y['update']['updates'],dense_collect=x['timing']['seconds'],compact_collect=y['timing']['seconds'],dense_update=x['update']['seconds'],compact_update=y['update']['seconds']))
 (R/f'resume-trajectory-{device}.json').write_text(json.dumps(dict(resume_bit_equal=True,keys=keys,comparison='4096+4 vs 4096+2+separate process+2, past exchange at4100',trajectory=rows),indent=2)+'\n')
 print(device,rows,flush=True)
