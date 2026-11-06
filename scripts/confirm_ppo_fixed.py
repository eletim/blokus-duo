import json,hashlib,sys
from pathlib import Path
import numpy as np
from blokus import ppo_gpu_eval as e
from blokus.ppo_runtime import seed_all
R=Path('artifacts/ppo-confirmation'); e.R=R;e.CACHE=Path('/tmp/ppo-lr64-training.npy')
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def run():
 seed_all(1784000)
 old=Path('artifacts/ppo-lr64')
 audit=json.loads((old/'training-audit.json').read_text());index=np.load(e.CACHE,mmap_mode='r')
 assert hashlib.sha256(index.tobytes()).hexdigest()==audit['union_sha256']
 write(R/'training-audit.json',audit)
 starts=set(json.loads((old/'excluded-starts.json').read_text())['states']);starts.update(p['state_hash'] for p in json.loads((old/'validation-prefixes.json').read_text())['prefixes'])
 write(R/'excluded-starts.json',dict(states=sorted(starts)))
 models={}
 for s in e.SEEDS:
  for c in ('B','C'):models[f'{c}-{s}']=str(old/f'{c}-seed-{s}'/'inference-5120.pt')
  models[f'gen2-{s}']=f'artifacts/ppo-gen2/seed-{s-1750000}.pt'
 lock={k:dict(path=p,sha256=e.sha(p)) for k,p in models.items()}
 if (R/'models.json').exists():assert json.loads((R/'models.json').read_text())==lock
 else:write(R/'models.json',lock)
 bank=e.bank('validation',256,17841000)
 groups={k:[] for k in ('B-gen2','C-gen2','C-B')}
 for label in groups:
  a,b=label.split('-')
  for s in e.SEEDS:
   result=e.match(models[f'{a}-{s}'],models[f'{b}-{s}'],bank,R/f'validation-{label}-{s}.json');groups[label].append((s,result));print(label,s,result['summary'],flush=True)
 summary={}
 for label,matches in groups.items():
  out=e.cluster(matches);by={p['id']:[] for p in bank['prefixes']}
  for s,m in matches:
   for g in m['games']:by[g['prefix_id']].append(g['difference'])
  point=np.array([np.mean(np.sign(v)*.5+.5) for v in by.values()]);ix=np.random.default_rng(1789900).integers(256,size=(20000,256));means=point[ix].mean(1)
  out['point_ci95']=np.quantile(means,[.025,.975]).tolist();out['point_familywise95_bonferroni']=np.quantile(means,[.05/6,1-.05/6]).tolist();out['superiority_adjusted']=out['point_familywise95_bonferroni'][0]>.5;summary[label]=out
 write(R/'summary.json',summary)
 e.audit_eval();a=json.loads((R/'evaluation-audit.json').read_text());assert all(x['later_state_overlap']==0 for x in a['rows'])
 print('AUDIT PASS',flush=True)
if __name__=='__main__':run()
