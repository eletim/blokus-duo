"""Read-only final aggregation; does not choose checkpoints or alter the gate."""
import gzip,json
from pathlib import Path
R=Path('artifacts/linear-tuple')
s=json.loads((R/'validation-summary.json').read_text());result={}
for pair,summary in s.items():
 left,right=pair.split('/');games=[]
 for seed in (3101,3102,3103):
  with gzip.open(R/f'validation-{left}-{right}-{seed}.json.gz','rt') as f:games+=json.load(f)['games']
 ds=[g['difference'] for g in games];result[pair]={**summary,'wins':sum(x>0 for x in ds),'draws':sum(x==0 for x in ds),'losses':sum(x<0 for x in ds)}
(R/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
