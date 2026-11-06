import subprocess,json,tempfile,sys
from pathlib import Path
R=Path('artifacts/linear-value');root=Path(tempfile.mkdtemp(prefix='linear-resume-'));rows=[]
for mode in ('old','new'):
 for name,end,resume in [('full',66,None),('split',63,None),('split',66,str(root/f'{mode}-split/state.json'))]:
  cmd=[sys.executable,'-m','blokus.linear_train2','--output',str(root/f'{mode}-{name}'),'--mode',mode,'--seed','9201','--until',str(end)]
  if resume:cmd+=['--resume',resume]
  subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
 a=json.loads((root/f'{mode}-full/state.json').read_text());b=json.loads((root/f'{mode}-split/state.json').read_text());assert a==b
 rows.append(dict(mode=mode,exact_json_equal=True,comparison='66 vs63+separate-process3, past swap64',policy_value_optimizer_rng=True))
(R/'resume-audit.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows)
