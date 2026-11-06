import json,collections
from pathlib import Path
import numpy as np
r=Path('artifacts/gen3-value-leaf');out={};wall=0
for p in sorted(r.glob('validation*-*-*-*.json')):
 row=json.loads(p.read_text());wall+=row['wall_seconds'];phase='validation2000' if p.name.startswith('validation2000') else 'validation500'
 for side,name in enumerate([row['left'],row['right']]):
  k=f"{phase}/{row['left']}-{row['right']}/{name}";o=out.setdefault(k,{'times':[],'nonpass':0,'forced_pass':0,'timeout_fallback':0,'depths':collections.Counter(),'root_widths':collections.Counter(),'nodes':0,'calls_policy':0,'calls_value':0,'cache_hits':0,'cold_load_seconds':[],'warmup_seconds':[]})
  o['cold_load_seconds'].append(row['cold_load_seconds']);o['warmup_seconds'].append(row['warmup_seconds'])
  for g in row['games']:
   for t in g['timings']:
    if t['side']!=side:continue
    o['times'].append(t['elapsed_ms']);o['forced_pass']+=t['forced_pass']
    if not t['forced_pass']:
     o['nonpass']+=1;o['timeout_fallback']+=t.get('fallback',False);o['depths'][t.get('depth',0)]+=1;o['root_widths'][t.get('root_width',0)]+=1
    for a,b in [('nodes','nodes'),('calls_policy','nn_policy_calls'),('calls_value','nn_value_calls'),('cache_hits','cache_hits')]:o[a]+=t.get(b,0)
for k,o in out.items():
 t=o.pop('times');budget=2000 if k.startswith('validation2000') else 500
 o.update(moves=len(t),mean_ms=float(np.mean(t)),p99_ms=float(np.quantile(t,.99)),max_ms=max(t),over_budget=sum(x>budget for x in t),over_budget_fraction=sum(x>budget for x in t)/len(t));o['depths']=dict(o['depths']);o['root_widths']=dict(o['root_widths'])
json.dump({'match_wall_seconds':wall,'rows':out,'unrecorded_old_A_calls':'not available, zero fields are not measured calls'},open(r/'timing-detail.json','w'),indent=2)
for k,o in out.items():print(k,o['nonpass'],o['timeout_fallback'],o['depths'],o['root_widths'])
