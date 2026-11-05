"""Finite serialized continuation after main export/integration, never overlap evaluation."""
import json,subprocess,sys,time
from pathlib import Path
r=Path('artifacts/ppo-4096')
if (r/'continuation-complete.json').exists():raise SystemExit('completed run is immutable')
assert (r/'main-integration.json').exists() and json.loads((r/'confirmation-audit.json').read_text())['pass_']
def run(cmd,log):
 with open(r/log,'w') as f:subprocess.run([sys.executable]+cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
run(['scripts/check_ppo_4096_resume.py'],'resume.log')
run(['-m','pytest','-q'],'tests.log')
for seed in (1401,1402,1403):
 used=sum(json.loads(p.read_text())['additional_seconds'] for p in r.glob('seed-*/result.json'));cap=min(1800,5400-used)
 if cap<=0:raise RuntimeError('continuation wall budget exhausted')
 run(['-m','blokus.ppo_scale','--seed',str(seed),'--resume',f'artifacts/ppo-learning-curve/seed-{seed}/iteration-2048.pt','--output',str(r/f'seed-{seed}'),'--iterations','4096','--save-steps','3072','4096','--cap',str(cap),'--forbidden-prefixes',str(r/'confirmation-prefixes.json')],f'train-{seed}.log')
for mode in ('audit_train','validation','holdout','audit_eval'):run(['-m','blokus.ppo_extend',mode],mode+'.log')
(r/'continuation-complete.json').write_text(json.dumps({'complete':True,'time':time.time()}))
