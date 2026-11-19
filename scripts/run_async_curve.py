"""Sequential development measurements, never overlapping heavy jobs."""
import subprocess,sys
from pathlib import Path
P=sys.executable;R=Path('artifacts/gpu-async512')
for n in (64,128,256,512):
 name=f'curve-{n}'
 if (R/name/'timing.json').exists():continue
 cmd=[P,'scripts/monitor_generation.py','--telemetry',str(R/f'{name}-telemetry.json'),P,'scripts/async_generation.py','--bank',str(R/'development-one-seed.json'),'--out',str(R/name),'--concurrency',str(n),'--fast-encoding','--graphs']
 with open(R/f'{name}.log','a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('finished',name,flush=True)
