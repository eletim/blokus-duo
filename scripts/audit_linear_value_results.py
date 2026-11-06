import json,hashlib
from pathlib import Path
import numpy as np
from scripts.linear_value_eval import cluster,model_sha
R=Path('artifacts/linear-value');checks=[]
for phase in ('validation','confirmation'):
 summary=json.loads((R/f'{phase}-summary.json').read_text());bank=json.loads((R/f'{phase}-prefixes.json').read_text())
 assert hashlib.sha256(json.dumps(bank['prefixes'],sort_keys=True).encode()).hexdigest()==bank['sha256']
 for label,expected in summary.items():
  left,right=label.split('/');matches=[json.loads((R/f'{phase}-{left}-{right}-{s}.json').read_text()) for s in (3101,3102,3103)]
  assert cluster(matches)==expected
  for d in matches:assert d['left_model_sha256']==model_sha(left,d['seed']) and d['right_model_sha256']==model_sha(right,d['seed'])
  checks.append(dict(phase=phase,pair=label,games=expected['games']))
lock=json.loads((R/'selection.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in lock['model_hashes'].items())
v=json.loads((R/'validation-summary.json').read_text());choices=[]
for budget in (6,25):
 label=f'search-{budget}/new'
 if label in v:
  stats=[]
  for s in (3101,3102,3103):stats.extend(json.loads((R/f'validation-search-{budget}-new-{s}.json').read_text())['left_search'])
  if v[label]['point']>=.5 and np.mean([s['fallback'] for s in stats])<.5:choices.append((v[label]['point'],-budget,f'search-{budget}'))
assert lock['search']==(max(choices)[2] if choices else None)
starts=lambda phase:{p['hash'] for p in json.loads((R/f'{phase}-prefixes.json').read_text())['prefixes']}
old=set(json.loads((R/'exclusions.json').read_text())['states']);assert not starts('validation')&old and not starts('confirmation')&(old|starts('validation'))
(R/'arithmetic-audit.json').write_text(json.dumps(dict(pass_=True,rows=checks,model_hashes=True,selection_rule=True,prefix_separation=True),indent=2)+'\n');print('ARITHMETIC PASS')
