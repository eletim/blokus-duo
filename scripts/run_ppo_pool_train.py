"""Six finite serialized runs. External comparison uses common completed steps."""
import sys,json,time,subprocess
from pathlib import Path
R=Path('artifacts/ppo-generation-pool')
assert json.loads((R/'cuda-resume.json').read_text())['pass_']
rows=[];used=0.
for seed in (1401,1402,1403):
 for kind in ('A','B'):
  out=R/kind/f'seed-{seed}';out.parent.mkdir(exist_ok=True)
  remaining=5400-used
  if remaining<120:break
  start=time.perf_counter()
  cmd=[sys.executable,'-m','blokus.ppo_scale','--seed',str(seed),'--resume',f'artifacts/ppo-6144/seed-{seed}/iteration-6144.pt','--output',str(out),'--iterations','7168','--save-steps','6400','6656','6912','7168','--cap',str(min(850,remaining-45)),'--forbidden-prefixes',str(R/'forbidden-prefixes.json'),'--pool-config',str(R/f'pool-{kind}.json')]
  with (R/f'train-{kind}-{seed}.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
  seconds=time.perf_counter()-start;used+=seconds;result=json.loads((out/'result.json').read_text());rows.append(dict(kind=kind,seed=seed,wall_seconds=seconds,result=result));(R/'budget.json').write_text(json.dumps(dict(limit=5400,total_seconds=used,rows=rows),indent=2));print(kind,seed,result,flush=True)
  if (out/'stopped.json').exists():raise RuntimeError('collapse diagnostic; do not extend')
assert used<=5400
(R/'training-complete.json').write_text(json.dumps(dict(time=time.time(),runs=len(rows),complete=len(rows)==6)))
