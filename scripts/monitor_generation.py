"""Record external utilization while one explicitly requested child runs."""
import argparse,subprocess,time,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--telemetry',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();start=time.perf_counter();rows=[]
child=subprocess.Popen(a.command)
try:
 while child.poll() is None:
  text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
  rows.append(dict(seconds=time.perf_counter()-start,gpu=text));time.sleep(1)
finally:
 Path(a.telemetry).write_text(json.dumps(dict(outer_seconds=time.perf_counter()-start,returncode=child.wait(),samples=rows),indent=2))
raise SystemExit(child.returncode)
