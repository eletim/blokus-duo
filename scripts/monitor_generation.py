"""Record external utilization while one explicitly requested child runs."""
import argparse,subprocess,time,json
import psutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--telemetry',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();start=time.perf_counter();rows=[]
child=subprocess.Popen(a.command)
try:
 while child.poll() is None:
  text=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).strip()
  stats=[]
  try:
   procs=[psutil.Process(child.pid)]+psutil.Process(child.pid).children(recursive=True)
   for proc in procs:
    try:
     t=proc.cpu_times();stats.append(dict(pid=proc.pid,cpu_seconds=t.user+t.system,rss=proc.memory_info().rss))
    except psutil.Error:pass
  except psutil.Error:pass
  rows.append(dict(seconds=time.perf_counter()-start,gpu=text,processes=stats));time.sleep(1)
finally:
 Path(a.telemetry).write_text(json.dumps(dict(outer_seconds=time.perf_counter()-start,returncode=child.wait(),samples=rows),indent=2))
raise SystemExit(child.returncode)
