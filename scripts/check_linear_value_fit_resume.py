import json,subprocess,sys,tempfile
from pathlib import Path
R=Path('artifacts/linear-value-retrain');rows=[]
with tempfile.TemporaryDirectory() as folder:
 for arm in ('P','S'):
  p=Path(folder);source='artifacts/linear-hybrid/exports/seed-3101.json';base=[sys.executable,'-m','blokus.linear_value_fit','--arm',arm,'--source',source,'--seed','9401']
  subprocess.run(base+['--until','4','--output',str(p/'whole.json')],check=True)
  subprocess.run(base+['--until','2','--output',str(p/'part.json')],check=True)
  subprocess.run([sys.executable,'-m','blokus.linear_value_fit','--resume',str(p/'part.json'),'--until','4','--output',str(p/'resume.json')],check=True)
  assert json.loads((p/'whole.json').read_text())==json.loads((p/'resume.json').read_text());rows.append(dict(arm=arm,comparison='4 vs 2+separate-process2',all_state_json_equal=True))
(R/'resume-audit.json').write_text(json.dumps(rows,indent=2)+'\n');print('RESUME PASS')
