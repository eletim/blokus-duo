"""Finite serialized pipeline; no generation/audit overlapping timed matches."""
import json,subprocess,sys,time
from pathlib import Path
r=Path('artifacts/ppo-6144')
if (r/'complete.json').exists():raise SystemExit('completed immutable run')
def run(cmd,log,timeout=None):
 start=time.perf_counter()
 with open(r/log,'w') as f:subprocess.run([sys.executable]+cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
 return time.perf_counter()-start
if '--resume-evaluation' in sys.argv:
 assert json.loads((r/'training-audit.json').read_text())['pass_']
 assert json.loads((r/'budget.json').read_text())['total_including_smoke_seconds']<=5400
 assert all(json.loads((r/f'seed-{s}/result.json').read_text())['complete'] for s in (1401,1402,1403))
 for mode in ('exclusion_audit','validation','holdout','audit_eval'):
  remaining=10800-(time.time()-json.loads((r/'started.json').read_text())['time'])
  if remaining<=0:raise RuntimeError('overall three-hour limit reached; keep partial results')
  run(['-m','blokus.ppo_6144_eval',mode],mode+'.log',min(3600,remaining))
 (r/'complete.json').write_text(json.dumps(dict(time=time.time(),complete=True,recovered_duplicate_audit=True)))
 raise SystemExit(0)
run(['-m','blokus.ppo_6144_eval','prepare'],'prepare.log')
smoke=run(['scripts/check_ppo_6144_resume.py'],'resume.log',300)
run(['-m','pytest','-q'],'tests.log',300)
used=smoke;walls=[]
for seed in (1401,1402,1403):
 remaining=5400-used;cap=min(1760,remaining-60)
 if cap<=0:raise RuntimeError('wall budget exhausted')
 wall=run(['-m','blokus.ppo_scale','--seed',str(seed),'--resume',f'artifacts/ppo-4096/seed-{seed}/iteration-4096.pt','--output',str(r/f'seed-{seed}'),'--iterations','6144','--save-steps','5120','6144','--cap',str(cap),'--forbidden-prefixes',str(r/'forbidden-prefixes.json')],f'train-{seed}.log',remaining)
 used+=wall;walls.append(dict(seed=seed,subprocess_seconds=wall));(r/'budget.json').write_text(json.dumps(dict(smoke_seconds=smoke,seeds=walls,total_including_smoke_seconds=used,limit=5400),indent=2))
 if (r/f'seed-{seed}/stopped.json').exists():raise RuntimeError('collapse guard; stop for diagnosis')
assert used<=5400
for mode in ('audit_train','validation','holdout','audit_eval'):run(['-m','blokus.ppo_6144_eval',mode],mode+'.log',3600)
(r/'complete.json').write_text(json.dumps(dict(time=time.time(),complete=True)))
