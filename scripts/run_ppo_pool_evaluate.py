"""Bounded continuation: wait for training exit, then audit/evaluate serially."""
import sys,json,time,subprocess
from pathlib import Path
R=Path('artifacts/ppo-generation-pool');deadline=json.loads((R/'started.json').read_text())['time']+10800
while not (R/'training-complete.json').exists():
 if time.time()>deadline:raise RuntimeError('three-hour limit: preserve partial run')
 time.sleep(5)
assert json.loads((R/'training-complete.json').read_text())['complete']
for mode in ('audit_train','validation','audit_eval','holdout','audit_eval'):
 if mode=='holdout' and not json.loads((R/'decision.json').read_text())['advance']:break
 remaining=deadline-time.time()
 if remaining<=0:raise RuntimeError('three-hour limit: preserve partial run')
 log=R/(('final-' if (R/'holdout-prefixes.json').exists() else '')+mode+'.log')
 with log.open('w') as f:subprocess.run([sys.executable,'-m','blokus.ppo_pool_eval',mode],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
(R/'complete.json').write_text(json.dumps(dict(complete=True,time=time.time(),holdout=bool((R/'holdout-prefixes.json').exists()))))
