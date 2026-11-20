import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
R=Path('artifacts/gpu-async512');m=json.loads((R/'final/measurement-manifest.json').read_text())
for path,h in m['source_shas'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
assert not (R/'steady-4096/timing.json').exists()
cmd=[sys.executable,'scripts/monitor_generation.py','--telemetry',str(R/'steady-4096-telemetry.json'),sys.executable,'scripts/async_generation.py','--bank',str(R/'steady-bank.json'),'--out',str(R/'steady-4096'),'--concurrency','512','--batch','128','--fast-encoding','--graphs']
start=time.perf_counter();timed_out=False
with (R/'steady-4096.log').open('w') as log:
 p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=1200)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=124
(R/'steady-run-status.json').write_text(json.dumps(dict(returncode=code,timed_out=timed_out,limit_seconds=1200,outer_seconds=time.perf_counter()-start),indent=2)+'\n')
raise SystemExit(code)
