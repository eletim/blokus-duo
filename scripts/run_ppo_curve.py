"""Finite serialized research pipeline. No learning/evaluation overlap."""
import json,subprocess,sys,time
from pathlib import Path
r=Path('artifacts/ppo-learning-curve')
if (r/'pipeline-complete.json').exists():raise SystemExit('completed experiment is immutable; use separate output directories for reproduction')
def run(cmd,log):
 with open(r/log,'w') as f:subprocess.run([sys.executable]+cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
# First seed was launched before this runner. Bounded wait, no orphan infinite process.
limit=time.monotonic()+1300
while not (r/'seed-1401/result.json').exists():
 if time.monotonic()>limit:raise TimeoutError('seed1401 did not finish')
 time.sleep(2)
for seed in (1402,1403):
 used=sum(json.loads(p.read_text())['additional_seconds'] for p in r.glob('seed-*/result.json'))
 cap=min(1200,3600-used)
 if cap<=0:raise RuntimeError('total continuation budget exhausted')
 run(['-m','blokus.ppo_scale','--seed',str(seed),'--resume',f'artifacts/terminal-ppo/extended/seed-{seed}/iteration-512.pt','--output',str(r/f'seed-{seed}'),'--cap',str(cap)],f'train-{seed}.log')
run(['scripts/check_ppo_scale_resume.py'],'resume.log')
run(['-m','pytest','-q'],'tests.log')
for mode in ('audit_train','dev','validation','holdout','audit_eval'):
 run(['-m','blokus.ppo_curve_eval',mode],mode+'.log')
(r/'pipeline-complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
