import sys,subprocess,json,hashlib
from pathlib import Path
P=sys.executable;R=Path('artifacts/gpu-async512/final')
commands=[('cpu-1',[P,'-c',f"import gpu_generation as g;from pathlib import Path;g.R=Path('{R}');g.benchmark('cpu',1)"]),
 ('async-1',[P,'scripts/async_generation.py','--bank',str(R/'bank.json'),'--out',str(R/'async-1'),'--concurrency','512','--graphs','--fast-encoding']),
 ('parallel-1',[P,'scripts/barrier_reference.py','--root',str(R),'--repeat','1']),
 ('async-2',[P,'scripts/async_generation.py','--bank',str(R/'bank.json'),'--out',str(R/'async-2'),'--concurrency','512','--graphs','--fast-encoding']),
 ('cpu-fast-1',[P,'scripts/cpu_fast_generation.py','--root',str(R),'--repeat','1']),
 ('cpu-2',[P,'-c',f"import gpu_generation as g;from pathlib import Path;g.R=Path('{R}');g.benchmark('cpu',2)"])]
for name,cmd in commands:
 manifest=json.loads((R/'measurement-manifest.json').read_text())
 for path,sha in manifest['source_shas'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
 done=R/name/'timing.json' if name.startswith('async') else R/f'{name}.json'
 if done.exists():print('already complete',name,flush=True);continue
 with open(R/f'{name}.log','w') as log:
  subprocess.run([P,'scripts/monitor_generation.py','--telemetry',str(R/f'{name}-telemetry.json')]+cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('finished',name,flush=True)
