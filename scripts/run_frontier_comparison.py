"""Serial measurement only. Persist GPU samples separately from work."""
import json,os,subprocess,sys,time
from pathlib import Path
R=Path('artifacts/gpu-frontier')
for mode,rep in [('cpu',1),('gpu',1),('gpu',2),('cpu',2)]:
 with (R/f'{mode}-{rep}.log').open('w') as f:
  p=subprocess.Popen([sys.executable,'scripts/frontier_generation.py',mode,'--repeat',str(rep)],stdout=f,stderr=subprocess.STDOUT)
  samples=[];start=time.time()
  while p.poll() is None:
   q=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
   samples.append(dict(t=time.time()-start,gpu=q.stdout.strip()));time.sleep(1)
  (R/f'{mode}-{rep}-gpu-samples.json').write_text(json.dumps(samples))
  if p.returncode:raise RuntimeError((mode,rep,p.returncode))
