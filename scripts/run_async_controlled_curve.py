"""One common frozen implementation for the final development concurrency curve."""
import hashlib,json,subprocess,sys
from pathlib import Path
P=sys.executable;R=Path('artifacts/gpu-async512');manifest=json.loads((R/'final/measurement-manifest.json').read_text())
for n in (512,256,128,64):
 for path,h in manifest['source_shas'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 name=f'controlled-{n}'
 if (R/name/'timing.json').exists():continue
 cmd=[P,'scripts/monitor_generation.py','--telemetry',str(R/f'{name}-telemetry.json'),P,'scripts/async_generation.py','--bank',str(R/'development-one-seed.json'),'--out',str(R/name),'--concurrency',str(n),'--fast-encoding','--graphs']
 with open(R/f'{name}.log','w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('finished',name,flush=True)
